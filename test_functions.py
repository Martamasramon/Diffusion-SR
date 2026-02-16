import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import csv
from datetime import datetime

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


def run_diffusion(diffusion, model_input, unet_type, controlnet=False, perform_uq=False, num_rep=None):
    """
    Unified diffusion sampling call.
    - lowres: the conditioned input (e.g., downsampled ADC)
    - t2w_input: optional extra conditioning (T2W) either as image tensor or embedding tuple/list
    - controlnet: toggles whether the conditioning is passed as `control` vs `t2w`
    """
    kwargs = {"batch_size": model_input['lowres'].shape[0]}
    # Add optional conditioning kwargs
    if model_input['t2w'] is not None:
        kwargs["control" if controlnet else "t2w"] = model_input['t2w']
    if model_input['hbv'] is not None:
        kwargs["hbv"] = model_input['hbv']
    kwargs['perform_uq'] = perform_uq  # Disable multiple reruns for standard sampling
    kwargs['num_rep']    = num_rep if perform_uq else None  # Number of samples for UQ
    
    # Sampling is inference-only
    with torch.no_grad():
        pred = diffusion.sample(model_input['lowres'], **kwargs)
        
    if unet_type == 'multitask':
        pred = pred[0] # pred[1] is t2w -> deal with this later

    return pred

def get_target_prediction(batch, model_output, vae=None):
    """
    Determine which target to use for evaluation and whether to transform the model output.
    - If 'ADC_target' exists: use it as target and downsample model_output to match.
    - Else: evaluate against 'ADC_input' directly.
    """
    if 'ADC_target' in batch.keys():
        # Build transform to match target resolution / shape
        pred_transform  = downsample_transform(batch['ADC_target'].shape[1])
        target     = batch['ADC_target']
        prediction = pred_transform(model_output)
    else:
        target      = batch['ADC_input']
        prediction  = model_output
    
    if vae is not None:
        prediction = decode_latent(prediction, vae)[:,0,:,:]
        
    return target, prediction

def get_batch_images(dataloader, device, use_T2W, use_HBV, vae=None, batch=None):
    """
    Fetch one batch from dataloader and move relevant tensors to device.
    Optionally:
      - encode lowres (and t2w) into latent space using a VAE
      - collect T2W conditioning if requested
    Returns:
      highres, lowres, t2w, hbv, batch
    """
    if batch is None:
        batch   = next(iter(dataloader))
    
    model_images = {
        'highres': batch['ADC_input'].to(device),
        'lowres':  batch['ADC_condition'].to(device),
        't2w_lowres':  None,
        't2w_highres': None,
        'hbv':     None
    }
    
    model_input = {
        'lowres': batch['ADC_condition'].to(device),
        't2w':    None,
        'hbv':    None
    }
            
    if vae is not None:
        model_input['lowres'], _ = encode_latent(model_input['lowres'], vae)
        
    if use_T2W:
        t2w = get_t2w_input(batch, device)
        model_input['t2w']  = t2w
        
        model_images['t2w_lowres']  = t2w
        model_images['t2w_highres'] = batch['T2W_input'].to(device)
        
        if vae is not None:
            model_input['t2w'],_ = encode_latent(model_input['t2w'], vae)        
        
    if use_HBV:
        hbv = batch['HBV'].to(device)
        model_input['hbv']  = hbv
        model_images['hbv'] = hbv
    
    return model_input, model_images, batch   

def log_metrics_to_csv(args, mse, psnr,ssim,csv_path='/cluster/project7/ProsRegNet_CellCount/CriDiff/results.csv'):
    """
    Append evaluation metrics + selected args to a CSV file.
    """
    row = { "timestamp": datetime.now().isoformat() }

    ARG_KEYS = [
        "checkpoint",
        "img_size",
        "down",
        "unet_type",
        "use_T2W",
        "use_HBV",
        "controlnet",
        "finetune"
    ]
    
    for k in ARG_KEYS:
        if hasattr(args, k):
            row[k] = getattr(args, k)
            
    row ["mse"]  = mse
    row ["psnr"] = psnr
    row ["ssim"] = ssim

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_results(args, diffusion, dataloader, device, vae=None):
    """
    Iterate over dataloader and compute average MSE/PSNR/SSIM over all samples.
    """
    mse_list, psnr_list, ssim_list = [], [], []
    
    for batch in dataloader:
        # Base conditioning input, Optional T2W for the diffusion model
        _, lowres, t2w_input, _ = get_batch_images(dataloader, device, args.use_T2W, vae, batch)

        # Sample SR output
        model_output  = run_diffusion(diffusion, lowres, t2w_input, args.unet_type, args.controlnet)
                    
        # Align prediction/target if necessary
        target, prediction = get_target_prediction(batch, model_output, vae)
        target = target.to(device)

        # Accumulate per-image metrics
        mse_list, psnr_list, ssim_list = add_batch_metrics_to_list(
            prediction, target, mse_list, psnr_list, ssim_list
        )
        
    mse  = np.mean(mse_list)
    psnr = np.mean(psnr_list)
    ssim = np.mean(ssim_list)  
    
    print(f'Average MSE:  {mse:.6f}')
    print(f'Average PSNR: {psnr:.2f}')
    print(f'Average SSIM: {ssim:.4f}')
    
    log_metrics_to_csv(args, mse, psnr, ssim)

def uq_calibration(x0_samples, highres):
    '''
    Compute the coverage of the 5-95% prediction interval.

    :param x0_samples: Tensor of shape (num_reruns, C, H, W) containing multiple rerun samples from the conditional predictive distribution. x0_samples are individial samples within a batch.
    :param highres: Tensor of shape (C, H, W) containing the ground truth high-resolution image.
    :return: Coverage value (float) representing the proportion of pixels in highres that fall
    '''
    lower_5 = torch.quantile(x0_samples, 0.05, dim=0) # lower 5th percentile; shape (C, H, W)
    upper_95 = torch.quantile(x0_samples, 0.95, dim=0) # upper 95th percentile; shape (C, H, W)

    within_interval = ((highres >= lower_5) & (highres <= upper_95)).float()
    coverage = within_interval.mean().item()  # Proportion of pixels within the interval

    return coverage


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
      - Spearman correlation coefficient
      - Overlay median error per predicted std quantile
    """
    pred_std_np = format_image(pred_std)
    err = np.abs(format_image(pred) - format_image(highres))

    # Scatter plot data
    pred_std_flat = np.asarray(pred_std_np.flatten())
    err_flat     = np.asarray(err.flatten())

    rho, p = spearmanr(pred_std_flat, err_flat)

    # Correlation scatter plot
    axes[i, j].scatter(pred_std_flat, err_flat, alpha=0.5)
    axes[i, j].set_xlabel('Predicted Std Deviation')
    axes[i, j].set_ylabel('Absolute Error')

    # Overlay median error per predicted std quantile
    n_bins = 10
    quantiles = np.quantile(pred_std_flat, np.linspace(0, 1, n_bins + 1))

    bin_centers = []
    bin_error_median = []

    for i in range(n_bins):
        mask = (pred_std_flat >= quantiles[i]) & (pred_std_flat < quantiles[i + 1])
        if np.any(mask):
            bin_centers.append(pred_std_flat[mask].mean())
            bin_error_median.append(np.median(err_flat[mask]))

    # Plot median error line
    axes[i, j].plot(
        bin_centers,
        bin_error_median,
        color="red",
        alpha=0.8,
        marker="o",
        linewidth=2,
        # label="Median error per std quantile"
    )

    # Annotate Spearman correlation coefficient at top-right in the plot
    axes[i, j].text(
        0.975, 0.975,
        rf"ρ={rho:.2f}",
        transform=axes[i, j].transAxes,
        ha='right',
        va='top',
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="none",
            alpha=0.25,
            edgecolor="black"
        )
    )
    # axes[i, j].set_title(f'Uncertainty-Error Correlation (Spearman ρ={rho:.2f})')

def plot_uq_t2w_overlay(t2w_img, pred_std, fig, axes, i, j, colorbar=True):
    """
    Plot T2W image with predicted std deviation overlay.
    """
    t2w_np     = format_image(t2w_img)
    pred_std_np = format_image(pred_std)

    axes[i, j].imshow(t2w_np, cmap='gray', vmin=0, vmax=1)
    im_overlay = axes[i, j].imshow(pred_std_np, cmap='hot', vmin=0, vmax=0.5, alpha=0.6)
    if colorbar:
        fig.colorbar(im_overlay, ax=axes[i, j])
    axes[i, j].axis('off')


def create_plot(batch_size, use_T2W, use_HBV, num_rep=None, offset=False, add_error=False, avg_std=False, uq_t2w_overlay=False):
    """
    Create a figure grid and set titles for the first row.
    Supports three modes:
      1) num_rep is None: single output visualization (optionally with T2W + error)
      2) num_rep set + offset=False: multiple SR samples for same input (optionally with T2W)
      3) num_rep set + offset=True: multiple pairs (T2W, SR) for per-rep T2W sampling
    """
    titles = ["Low res (Input)"]

    if use_HBV:
        titles += ["HBV (Input)"]
    
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

        if uq_t2w_overlay:
            titles += ["T2W with Uncertainty Overlay"]

        if add_error:
            titles += ["Uncertainty-Error Correlation"]
    
    ncols = len(titles)
    print(titles)

    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols, 3*batch_size))
    for j, t in enumerate(titles):
        axes[0, j].set_title(t)
        
    return fig, axes, ncols
     

def decode_all_UQ(pred,vae):
    if vae is not None:
        decoded_x0 = []
        for i in range(pred.shape[1]):
            decoded_x0_sample = decode_latent(pred[:,i,:,:,:], vae)[:,0,:,:]
            decoded_x0.append(decoded_x0_sample)
        decoded_x0 = torch.stack(decoded_x0, dim=1)  # Shape: (batch_size, num_reruns, H, W)

        pred_mean = decoded_x0.mean(dim=1)  # Use mean prediction for visualization; shape: (batch_size, H, W)
        pred_std  = decoded_x0.std(dim=1)   # Use std for uncertainty maps; shape: (batch_size, H, W)
    
    else:
        decoded_x0 = pred 
        pred_mean = pred.mean(dim=1)  # Use mean prediction for visualization; shape: (batch_size, H, W)
        pred_std  = pred.std(dim=1)
    
    return decoded_x0, pred_mean, pred_std

def visualize_batch(
    args, diffusion, dataloader, device,
    output_name="test_image", vae=None, add_error=True, perform_uq=False, num_rep=None
):
    """
    Visualize a single batch:
      - lowres input
      - optional T2W input
      - SR prediction
      - optional error map
      - highres ground truth
    """
    if num_rep is not None:
        avg_std, add_error = True, True    
    else:
        avg_std = False

    uq_t2w_overlay = False # Set to False for now; could be set to True if the overlay plot is desired after enabling T2W input

    fig, axes, ncols = create_plot(args.batch_size, args.use_T2W, args.use_HBV, num_rep=num_rep, add_error=add_error, avg_std=avg_std, uq_t2w_overlay=uq_t2w_overlay)
    model_input, model_images, batch = get_batch_images(dataloader, device, args.use_T2W, args.use_HBV, vae)
    
    # Run model sampling
    pred = run_diffusion(diffusion, model_input, args.unet_type, args.controlnet, perform_uq, num_rep)
    
    # Uncertainty quantification
    if perform_uq:
        decoded_x0, pred_mean, pred_std = decode_all_UQ(pred, vae)
    elif vae is not None:
        pred = decode_latent(pred, vae)[:,0,:,:]
    
    for i in range(args.batch_size):
        count = 0
        # Column 0: lowres input
        plot_image(model_images['lowres'][i], fig, axes, i, 0)

        # OPTIONAL: HBV input
        if args.use_HBV:
            count += 1
            plot_image(model_images['hbv'][i], fig, axes, i, count)
        
        if num_rep is None:
            # Column 1 (optional): T2W input
            if args.use_T2W:
                count += 1                    
                if "T2W_embed" in batch:
                    plot_image(model_images['t2w_lowres'][0][i], fig, axes, i, 1)
                else:
                    plot_image(model_images['t2w_lowres'][i], fig, axes, i, 1)
                    
            # Column 2: High res (SR Output)
            plot_image(pred[i], fig, axes, i, count+1)
            
            # Column 3 (optional): Error
            if add_error:
                plot_image(pred[i], fig, axes, i, count+2, False)
                plot_error(pred[i], model_images['highres'][i], fig, axes, i, count+2)
                count += 2
                
        else:
            if args.use_T2W:
                count += 1
                # Add t2w_embed stuff?
                plot_image(model_images['t2w_lowres'][i], fig, axes, i, 1)
                    
            # Columns (x num_rep): High res (SR Outputs)
            for rep in range(num_rep):
                count += 1
                plot_image(decoded_x0[i][rep], fig, axes, i, count)

            plot_image(pred_mean[i], fig, axes, i, count+1)
            plot_image(pred_std[i],  fig, axes, i, count+2, kind='std')
            count += 2

        # Ground truth (last column)
        plot_image(model_images['highres'][i], fig, axes, i, count+1)

        # T2W with uncertainty overlay (optional)
        if args.use_T2W and uq_t2w_overlay:
            plot_uq_t2w_overlay(model_images['t2w_highres'][i], pred_std[i], fig, axes, i, ncols-2)
    
        # UQ error column (optional)
        if add_error and num_rep is not None:
            plot_uq_error_corr(pred_mean[i], model_images['highres'][i], pred_std[i], fig, axes, i, ncols-1)

        # # Calibration coverage of gt HR ADC within x0_samples (optional)
        # coverage_x0_gt = uq_calibration(decoded_x0[i], highres[i])
        # print(f"Batch SR Image {i}: Coverage of 5-95% PI: {coverage_x0_gt*100:.2f}%")
    
    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")


# -------------------------------------------------
# -------------------------------------------------

def visualize_variability(
    args, diffusion, dataloader, device, 
    output_name="test_image", num_rep=5, avg_std=False, vae=None
):
    """
    Visualize variability by sampling multiple SR outputs for the same input.
    Optionally show mean/std over repetitions.
    """
    fig, axes, ncols = create_plot(args.batch_size, args.use_T2W, num_rep=num_rep, offset=False, add_error=False, avg_std=avg_std)
    highres, lowres, t2w_input, _ = get_batch_images(dataloader, device, args.use_T2W, vae)
    
    # Collect multiple stochastic samples
    all_pred = []
    for rep in range(num_rep):
        pred = run_diffusion(diffusion, lowres, t2w_input, args.unet_type, args.controlnet)
        all_pred.append(format_image(pred))
        
    all_pred = np.array(all_pred)

    # Optional summary statistics over reps
    if avg_std:
        mean_pred = np.mean(all_pred, axis=0)                 
        std_pred  = np.std(all_pred, axis=0)

    count = 1 if args.use_T2W else 0
    for i in range(args.batch_size):
        # Column 0: lowres
        plot_image(lowres[i], fig, axes, i, 0)
        # Column 1 (optional): T2W
        if args.use_T2W:
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")


def visualize_variability_t2w(
    args, diffusion, dataloader, device, 
    output_name="test_image", num_rep=5, avg_std=False
):
    """
    Visualize variability when the T2W input itself can vary per repetition
    (e.g., random augmentations or sampling along the T2W pipeline).
    For each rep:
      - load T2W from file paths
      - apply dataset transform
      - sample diffusion output conditioned on that T2W
    """
    fig, axes, ncols = create_plot(args.batch_size, True, num_rep=num_rep, offset=True, add_error=False, avg_std=avg_std)
    highres, lowres, _, batch = get_batch_images(dataloader, device, args.use_T2W)
    
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
        pred = run_diffusion(diffusion, lowres, t2w_input, args.unet_type, args.controlnet)
                
        # Store for later plotting
        all_pred.append(format_image(pred))
        all_t2w.append(format_image(t2w_input))
        
    all_pred = np.array(all_pred)
    all_t2w  = np.array(all_t2w)

    # Optional summary stats on outputs
    if avg_std:
        mean_pred = np.mean(all_pred, axis=0)
        std_pred  = np.std(all_pred, axis=0)

    for i in range(args.batch_size):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

        
    
    