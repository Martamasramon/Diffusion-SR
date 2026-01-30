import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import cv2 
from torchvision     import transforms as T
from skimage.metrics import structural_similarity   as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import mean_squared_error      as mse_metric
from transforms import downsample_transform
from PIL import Image
from scipy.stats import spearmanr

import sys
sys.path.append('../models')
from models.network_utils import *
from VAE import encode_latent, decode_latent


def compute_metrics(pred, gt):
    """
    Compute per-image metrics between prediction and ground truth.
    Assumes tensors are in [0,1] range for PSNR/SSIM data_range=1.0.
    """
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()
    
    mse  = mse_metric (gt_np, pred_np)
    psnr = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim = ssim_metric(gt_np, pred_np, data_range=1.0)
    return mse, psnr, ssim


def get_t2w_input(batch, device):
    """
    Extract T2W input from the batch and move it to the correct device.
    Supports either:
      - 'T2W_embed' (tuple/list of multiple tensors used as an embedding input)
      - 'T2W_condition' (image-like tensor conditioning)
    """
    if "T2W_embed" in batch:
        # Expecting multiple tensors; squeeze/reshape first element to include a channel dim
        batch_squeezed   = [np.squeeze(i) for i in batch['T2W_embed']]
        batch_final_dims = (
            batch_squeezed[0].unsqueeze(1),
            batch_squeezed[1],
            batch_squeezed[2],
            batch_squeezed[3]
        )
        # Move each element to device
        output = [i.to(device) for i in batch_final_dims]
    else:
        # Image-like T2W conditioning
        batch_final_dims = batch['T2W_condition']
        output = batch_final_dims.to(device)
        
    return output


def add_batch_metrics_to_list(prediction, highres, mse_list, psnr_list, ssim_list):
    """
    Loop over batch dimension and append metrics for each pair (pred, gt).
    """
    for j in range(prediction.size(0)):
        mse, psnr, ssim = compute_metrics(prediction[j], highres[j])
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    return mse_list, psnr_list, ssim_list


def run_diffusion(diffusion, lowres, t2w_input, controlnet=False, perform_uq=False, num_rep=None):
    """
    Unified diffusion sampling call.
    - lowres: the conditioned input (e.g., downsampled ADC)
    - t2w_input: optional extra conditioning (T2W) either as image tensor or embedding tuple/list
    - controlnet: toggles whether the conditioning is passed as `control` vs `t2w`
    """
    kwargs = {"batch_size": lowres.shape[0]}
    # Add optional conditioning kwargs
    if t2w_input is not None:
        kwargs["control" if controlnet else "t2w"] = t2w_input
    kwargs['perform_uq'] = perform_uq  # Disable multiple reruns for standard sampling
    kwargs['num_rep']    = num_rep if perform_uq else None  # Number of samples for UQ
    
    # Sampling is inference-only
    with torch.no_grad():
        return diffusion.sample(lowres, **kwargs)


def get_target_prediction(batch, model_output):
    """
    Determine which target to use for evaluation and whether to transform the model output.
    - If 'ADC_target' exists: use it as target and downsample model_output to match.
    - Else: evaluate against 'ADC_input' directly.
    """
    if 'ADC_target' in batch.keys():
        # Build transform to match target resolution / shape
        pred_transform  = downsample_transform(batch['ADC_target'].shape[1])
        return batch['ADC_target'], pred_transform(model_output)
    else:
        return batch['ADC_input'], model_output


def evaluate_results(diffusion, dataloader, device, batch_size, use_T2W=False, controlnet=False):
    """
    Iterate over dataloader and compute average MSE/PSNR/SSIM over all samples.
    """
    mse_list, psnr_list, ssim_list = [], [], []
    
    for batch in dataloader:
        # Base conditioning input, Optional T2W for the diffusion model
        adc_condition = batch['ADC_condition'].to(device)
        t2w_input     = get_t2w_input(batch, device) if use_T2W else None

        # Sample SR output
        model_output  = run_diffusion(diffusion, adc_condition, t2w_input, controlnet)
                    
        # Align prediction/target if necessary
        target, prediction = get_target_prediction(batch, model_output)
        target = target.to(device)

        # Accumulate per-image metrics
        mse_list, psnr_list, ssim_list = add_batch_metrics_to_list(
            prediction, target, mse_list, psnr_list, ssim_list
        )
        
    print(f'Average MSE:  {np.mean(mse_list):.6f}')
    print(f'Average PSNR: {np.mean(psnr_list):.2f}')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')


def format_image(x):
    """
    Standardize input to a 2D numpy array for imshow:
    - tensor -> numpy
    - squeeze singleton dims (e.g., [1,H,W] -> [H,W])
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.squeeze(x)


def plot_image(image, fig, axes, i, j, colorbar=False, kind='default'):
    """
    Plot a single image into axes[i,j], with optional colour scale presets:
      - kind='default': vmin=0, vmax=1
      - kind='std':     vmin=0, vmax=0.5 (for variability/std maps)
      - kind='vae':     vmin=-1, vmax=1-ish (you set vmin=-1; vmax kept at 1)
    """
    vmin, vmax = 0, 1 
    if kind=='std':
        vmax = 0.5 
    if kind=='vae':
        vmin = -1
        
    img_plot = axes[i, j].imshow(format_image(image), cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, j].axis('off')
    if colorbar:
        fig.colorbar(img_plot, ax=axes[i, j])


def plot_error(pred, highres, fig, axes, i, j):
    """
    Plot normalized absolute error overlay in axes[i,j].
    Normalization uses a high percentile to reduce sensitivity to outliers.
    """
    err = np.abs(format_image(pred) - format_image(highres))

    # Robust normalization (avoid blowing up on rare spikes)
    p99 = np.percentile(err, 99.5)
    den = p99 if p99 > 1e-8 else (err.max() + 1e-8)
    err_norm = np.clip(err / den, 0, 1)

    im_overlay = axes[i, j].imshow(err_norm, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.6)
    cbar = fig.colorbar(im_overlay, ax=axes[i, j])

def plot_uq_error_corr(pred, highres, pred_std, fig, axes, i=None, j=None):
    """
    Plot uncertainty quantification results:
      - Absolute error between pred and highres
      - Predicted std deviation map
      - Correlation scatter plot between error and predicted std
    """
    err = np.abs(format_image(pred) - format_image(highres))
    pred_std_np = format_image(pred_std)

    # Scatter plot data
    err_flat     = err.flatten()
    pred_std_flat = pred_std_np.flatten()

    rho, p = spearmanr(err_flat, pred_std_flat)

    # Correlation scatter plot
    axes[i, j].scatter(pred_std_flat, err_flat, alpha=0.5)
    axes[i, j].set_xlabel('Predicted Std Deviation')
    axes[i, j].set_ylabel('Absolute Error')
    axes[i, j].set_title(f'Uncertainty-Error Correlation (Spearman $\rho$={rho:.2f})')


def create_plot(batch_size, use_T2W, num_rep=None, offset=False, add_error=False, avg_std=False):
    """
    Create a figure grid and set titles for the first row.
    Supports three modes:
      1) num_rep is None: single output visualization (optionally with T2W + error)
      2) num_rep set + offset=False: multiple SR samples for same input (optionally with T2W)
      3) num_rep set + offset=True: multiple pairs (T2W, SR) for per-rep T2W sampling
    """
    titles = ["Low res (Input)"]
    
    if num_rep is None: # Single SR output case
        if use_T2W:
            titles += ["High res T2W (Input)"]
        titles += ["High res (SR Output)"]
        if add_error:
            titles += ["Error"]
        titles += ["High res (Ground truth)"]

    else: # Variability / multiple samples case
        if offset: # Each rep has a (T2W, SR) pair
            for r in range(num_rep):
                titles += ["High res T2W", f"Super resolution ({r+1})"]
        else: # One T2W input, multiple SR outputs
            if use_T2W:
                titles += ["High res T2W (Input)"]
            titles += [f"Super resolution ({r+1})" for r in range(num_rep)]

        # Optional mean/std summary columns
        if avg_std:
            titles += ["Average Output", "Std Output"]

        titles += ["High res (Ground truth)"]

        if add_error and not avg_std:
            titles += ["Uncertainty-Error Correlation"]
    
    ncols = len(titles)

    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols, 3*batch_size))
    for j, t in enumerate(titles):
        axes[0, j].set_title(t)
        
    return fig, axes, ncols


def get_batch_images(dataloader, device, use_T2W, vae=None, ):
    """
    Fetch one batch from dataloader and move relevant tensors to device.
    Optionally:
      - encode lowres (and t2w) into latent space using a VAE
      - collect T2W conditioning if requested
    Returns:
      highres, lowres, t2w_input, batch
    """
    batch   = next(iter(dataloader))
    highres = batch['ADC_input'].to(device)
    lowres  = batch['ADC_condition'].to(device)
    
    if vae is not None:
        lowres, _ = encode_latent(lowres, vae)
        
    if use_T2W:
        t2w_input = get_t2w_input(batch, device)
        if vae is not None:
            t2w_input,_ = encode_latent(t2w_input, vae)
    else:
        t2w_input = None
    
    return highres, lowres, t2w_input, batch        


def visualize_batch(
    diffusion, dataloader, batch_size, device,
    use_T2W=False, controlnet=False, output_name="test_image", vae=None, add_error=True, perform_uq=False, num_rep=None
):
    """
    Visualize a single batch:
      - lowres input
      - optional T2W input
      - SR prediction
      - optional error map
      - highres ground truth
    """
    avg_std = False if not perform_uq else True
    num_rep = num_rep if perform_uq else None
    fig, axes, ncols = create_plot(batch_size, use_T2W, num_rep=num_rep, add_error=add_error, avg_std=avg_std)
    highres, lowres, t2w_input, batch = get_batch_images(dataloader, device, use_T2W, vae)
    
    # Run model sampling
    if diffusion is not None:
        pred = run_diffusion(diffusion, lowres, t2w_input, controlnet, perform_uq, num_rep)
        
    # else:
    #     lowres    = batch['ADC_condition']
    #     full_size = lowres.shape[-1]
    #     for i in range(batch_size):
    #         lowres[i] = cv2.resize(lowres[i], (full_size//2,full_size//2))
    #         lowres[i] = cv2.resize(lowres[i], (full_size,full_size), interpolation=cv2.INTER_LINEAR)
    #     lowres    = lowres.to(device)
    
    # Optional: decode from latent to pixel space for visualization
    pred, pred_std = None, None
    decoded_x0_samples = []
    if vae is not None:
        if diffusion is not None:
            if not perform_uq:
                pred = decode_latent(pred, vae)
            else:
                # Decode each sample in the UQ output
                for i in range(pred.shape[1]):
                    decoded_x0_sample = decode_latent(pred[:,i,:,:,:], vae)
                    decoded_x0_samples.append(decoded_x0_sample)
                decoded_x0_samples = torch.stack(decoded_x0_samples, dim=1)  # Shape: (batch_size, num_reruns, C, H, W)
                pred = decoded_x0_samples.mean(dim=1)  # Use mean prediction for visualization; shape: (batch_size, C, H, W)
                pred_std = decoded_x0_samples.std(dim=1)  # Use std for uncertainty maps; shape: (batch_size, C, H, W)
        else:
            pred = decode_latent(lowres, vae)
        pred   = pred[:,0,:,:]
        if pred_std is not None:
            pred_std = pred_std[:,0,:,:]
        lowres = batch['ADC_condition'].to(device)
    
    elif perform_uq:
        for i in range(pred.shape[1]):
            decoded_x0_sample = pred[:,i,:,:,:]
            decoded_x0_samples.append(decoded_x0_sample)
        decoded_x0_samples = torch.stack(decoded_x0_samples, dim=1)  # Shape: (batch_size, num_reruns, C, H, W)
        pred = decoded_x0_samples.mean(dim=1)  # Use mean prediction for visualization; shape: (batch_size, C, H, W)
        pred_std = decoded_x0_samples.std(dim=1)  # Use std for uncertainty maps; shape: (batch_size, C, H, W)
    
    for i in range(batch_size):
        count = 0
        # Column 0: lowres input
        plot_image(lowres[i], fig, axes, i, 0)
        # Column 1 (optional): T2W input
        if use_T2W:
            count += 1
            if "T2W_embed" in batch:
                # If using latent embedding, decode for plotting (if applicable)
                if vae is not None:
                    t2w_input = decode_latent(t2w_input, vae)[:,0,:,:]
                plot_image(t2w_input[i], fig, axes, i, 1)
            else:
                plot_image(t2w_input[0][i], fig, axes, i, 1)

        # SR output column
        if not perform_uq:
            count += 1
            plot_image(pred[i], fig, axes, i, count)

        if avg_std and perform_uq:
            # Repetition columns: SR outputs
            for rep in range(num_rep):
                count += 1
                plot_image(decoded_x0_samples[i][rep], fig, axes, i, count)
            # mean/std columns (optional; only if UQ performed)
            plot_image(pred[i], fig, axes, i, count+1)
            plot_image(pred_std[i],  fig, axes, i, count+2, kind='std')
            count += 2

        # Error column (optional)
        if add_error:
            plot_image(pred[i], fig, axes, i, count+1, False)
            plot_error(pred[i], highres[i], fig, axes, i, count+2)
            count += 2
        # Ground truth (last column)
        plot_image(highres[i], fig, axes, i, count+1)
        count += 1

        if add_error and not avg_std:
            # Plot UQ error correlation if applicable
            plot_uq_error_corr(pred[i], highres[i], pred_std[i], fig, axes, i, count+1)
            count += 1
    
    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")


def visualize_variability(
    diffusion, dataloader, batch_size, device,
    use_T2W=False, controlnet=False,
    output_name="test_image", num_rep=5, avg_std=False, vae=None
):
    """
    Visualize variability by sampling multiple SR outputs for the same input.
    Optionally show mean/std over repetitions.
    """
    fig, axes, ncols = create_plot(batch_size, use_T2W, num_rep=num_rep, offset=False, add_error=False, avg_std=avg_std)
    highres, lowres, t2w_input, _ = get_batch_images(dataloader, device, use_T2W, vae)
    
    # Collect multiple stochastic samples
    all_pred = []
    for rep in range(num_rep):
        pred = run_diffusion(diffusion, lowres, t2w_input, controlnet)
        all_pred.append(format_image(pred))
        
    all_pred = np.array(all_pred)

    # Optional summary statistics over reps
    if avg_std:
        mean_pred = np.mean(all_pred, axis=0)                 
        std_pred  = np.std(all_pred, axis=0)

    count = 1 if use_T2W else 0
    for i in range(batch_size):
        # Column 0: lowres
        plot_image(lowres[i], fig, axes, i, 0)
        # Column 1 (optional): T2W
        if use_T2W:
            plot_image(t2w_input[i], fig, axes, i, 1)
        # Repetition columns: SR outputs
        for rep in range(num_rep):
            plot_image(all_pred[rep][i], fig, axes, i, rep+count+1)
        # Optional mean/std columns
        if avg_std:
            plot_image(mean_pred[i], fig, axes, i, ncols-3)
            plot_image(std_pred[i],  fig, axes, i, ncols-2, kind='std')
        # Final column: ground truth
        plot_image(highres[i], fig, axes, i, ncols-1)

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'_variability.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")


def visualize_variability_t2w(
    diffusion, dataloader, batch_size, device,
    controlnet=False, output_name="test_image",
    num_rep=5, avg_std=False
):
    """
    Visualize variability when the T2W input itself can vary per repetition
    (e.g., random augmentations or sampling along the T2W pipeline).
    For each rep:
      - load T2W from file paths
      - apply dataset transform
      - sample diffusion output conditioned on that T2W
    """
    fig, axes, ncols = create_plot(batch_size, True, num_rep=num_rep, offset=True, add_error=False, avg_std=avg_std)
    highres, lowres, _, batch = get_batch_images(dataloader, device, use_T2W=False)
    
    all_pred = []
    all_t2w  = []
    
    # Use dataset's T2W transform to match training preprocessing
    t2w_transform = dataloader.dataset.transforms['T2W_condition']

    for rep in range(num_rep):
        # Load and transform T2W images per sample in the batch
        t2w_batch = []
        for p in batch['T2W_path']:
            t2w_img = Image.open(p).convert('L')
            t2w_batch.append(t2w_transform(t2w_img))
        t2w_input = torch.stack(t2w_batch, dim=0).to(device)
        
        # Sample conditioned output
        pred = run_diffusion(diffusion, lowres, t2w_input, controlnet)
                
        # Store for later plotting
        all_pred.append(format_image(pred))
        all_t2w.append(format_image(t2w_input))
        
    all_pred = np.array(all_pred)
    all_t2w  = np.array(all_t2w)

    # Optional summary stats on outputs
    if avg_std:
        mean_pred = np.mean(all_pred, axis=0)
        std_pred  = np.std(all_pred, axis=0)

    for i in range(batch_size):
        # Column 0: lowres
        plot_image(lowres[i], fig, axes, i, 0)
        # For each rep: (T2W, pred) columns
        for rep in range(num_rep):
            plot_image(all_t2w[rep][i], fig, axes, i, 1+rep*2)
            plot_image(all_pred[rep][i], fig, axes, i, 2+rep*2)
        # Optional mean/std (at the end)
        if avg_std:
            plot_image(mean_pred[i], fig, axes, i, ncols-3)
            plot_image(std_pred[i],  fig, axes, i, ncols-2, kind='std')
        # Final column: ground truth
        plot_image(highres[i], fig, axes, i, ncols-1)

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'_variability_t2w.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

    
def visualize_batch_vae(vae, dataloader, accelerator, output_name, greyscale=False):
    """
    Visualize VAE reconstructions on a single batch.
    - Uses T2W_condition if present, else falls back to ADC_input.
    - input_to_shape(...) is used to standardize dimensions/channels.
    """
    batch = next(iter(dataloader))
    x = batch.get("T2W_condition", batch["ADC_input"])
    x = x.to(accelerator.device)
    x = input_to_shape(x, greyscale)
    batch_size = x.size(0)
    
    ncols = 2
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols, 3*batch_size))
    axes[0,0].set_title('Input')
    axes[0,1].set_title('Output 0')
    # axes[0,2].set_title('Output 1')
    # axes[0,3].set_title('Output 2')
    # axes[0,2].set_title('Error')
    
    # Encode -> decode
    z, posterior = encode_latent(x, vae)
    x_recon = vae.decode(z)

    for i in range(batch_size):
        # Plot input and reconstruction (VAE typically outputs in [-1,1] depending on training)
        plot_image(x[i][0],       fig, axes, i, 0, kind='vae')
        plot_image(x_recon[i][0], fig, axes, i, 1, kind='vae')
        # plot_image(x_recon[i][1], fig, axes, i, 2)
        # plot_image(x_recon[i][2], fig, axes, i, 3)
        # plot_image(x_recon[i][0], fig, axes, i, 2, False)
        # plot_error(x_recon[i][0], x[i][0], fig, axes, i, 2)

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

        
    
    