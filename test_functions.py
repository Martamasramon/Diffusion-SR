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

import sys
sys.path.append('../models')
from models.network_utils import *
from VAE import encode_latent, decode_latent


def compute_metrics(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()
    
    mse  = mse_metric (gt_np, pred_np)
    psnr = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim = ssim_metric(gt_np, pred_np, data_range=1.0)
    return mse, psnr, ssim

def adapt_input_dims(batch):
    try:
        batch_squeezed   = [np.squeeze(i) for i in batch['T2W_embed']]
        batch_final_dims = (batch_squeezed[0].unsqueeze(1), batch_squeezed[1], batch_squeezed[2], batch_squeezed[3])
    except:
        batch_final_dims = batch['T2W_condition']
    
    return batch_final_dims

def to_device(t2w_input, device):
    try:
        t2w_input = t2w_input.to(device)
    except:
        t2w_input = [i.to(device) for i in t2w_input]
    return  t2w_input
 
def add_batch_metrics_to_list(prediction, highres, mse_list, psnr_list, ssim_list):
    for j in range(prediction.size(0)):
        mse, psnr, ssim = compute_metrics(prediction[j], highres[j])
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    return mse_list, psnr_list, ssim_list

def run_diffusion_with_t2w(batch, device, diffusion, lowres, controlnet=False):
    t2w_input     = adapt_input_dims(batch)
    t2w_input_gpu = to_device(t2w_input, device)
    
    with torch.no_grad():
        if controlnet:
            return diffusion.sample(lowres, batch_size=lowres.shape[0], control=t2w_input_gpu)
        else:
            return diffusion.sample(lowres, batch_size=lowres.shape[0], t2w=t2w_input_gpu)
                
def run_diffusion_without_t2w(batch, device, diffusion, lowres):
    with torch.no_grad():
        return diffusion.sample(lowres, batch_size=lowres.shape[0])
    
def get_target_prediction(batch, model_output):
    if 'ADC_target' in batch.keys():
        # Check this gets correct dims
        pred_transform  = downsample_transform(batch['ADC_target'].shape[1])
        return batch['ADC_target'], pred_transform(model_output)
    else:
        return batch['ADC_input'], model_output
    
def evaluate_results(diffusion, dataloader, device, batch_size, use_T2W=False, controlnet=False):
    mse_list, psnr_list, ssim_list = [], [], []
    
    for batch in dataloader:
        adc_condition = batch['ADC_condition'].to(device)
        
        if use_T2W:
            model_output = run_diffusion_with_t2w(batch, device, diffusion, adc_condition, controlnet)
        else:
            model_output = run_diffusion_without_t2w(batch, device, diffusion, adc_condition)
            
        target, prediction = get_target_prediction(batch, model_output)
        target = target.to(device)

        mse_list, psnr_list, ssim_list = add_batch_metrics_to_list(prediction, target, mse_list, psnr_list, ssim_list)
        
    print(f'Average MSE:  {np.mean(mse_list):.6f}')
    print(f'Average PSNR: {np.mean(psnr_list):.2f}')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')


def format_image(tensor):
    # Convert from torch tensor [1, H, W] -> [H, W], ensure it's on CPU and numpy
    return tensor.squeeze().detach().cpu().numpy()
    
def plot_image(image, fig, axes, i, j, colorbar=True, std=False):
    try:
        image  = format_image(image)
    except:
        pass
       
    vmax = 0.5 if std else 1
    img_plot = axes[i, j].imshow(image,  cmap='gray', vmin=0, vmax=vmax)
                                     
    axes[i, j].axis('off')
    # if colorbar:
    #     fig.colorbar(img_plot, ax=axes[i, j])
    
def plot_error(pred, highres, fig, axes, i, j):
    # Error
    err = np.abs(format_image(pred) - format_image(highres))
    p99 = np.percentile(err, 99.5)
    den = p99 if p99 > 1e-8 else (err.max() + 1e-8)
    err_norm = np.clip(err / den, 0, 1)

    im_overlay = axes[i, j].imshow(err_norm, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.6)
    cbar = fig.colorbar(im_overlay, ax=axes[i, j])
    
def visualize_batch(diffusion, dataloader, batch_size, device, use_T2W=False, controlnet=False, output_name="test_image", vae=None):
    ncols = 5 if use_T2W else 4
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    axes[0,0].set_title('Low res (Input)')
    axes[0,1].set_title('Super resolution (Output)')
    axes[0,2].set_title('Error')
    axes[0,3].set_title('High res (Ground truth)')
    if use_T2W:
        axes[0,4].set_title('High res T2W')
        
    # Get images 
    batch   = next(iter(dataloader))
    highres = batch['ADC_input'].to(device)
    lowres  = batch['ADC_condition'].to(device)
    
    if vae is not None:
        lowres, _ = encode_latent(lowres, vae)
    
    if diffusion is not None:
        if use_T2W:
            try:
                t2w_input = [np.squeeze(i).to(device) for i in batch['T2W_embed']]
                t2w_input = (t2w_input[0].unsqueeze(1), t2w_input[1], t2w_input[2], t2w_input[3])
                t2w_image = False
            except:
                t2w_input = batch['T2W_condition'].to(device)
                t2w_image = True
                
            if vae is not None:
                t2w_input,_ = encode_latent(t2w_input, vae)

            with torch.no_grad():
                if controlnet:
                    pred  = diffusion.sample(lowres, batch_size=lowres.shape[0], control=t2w_input)
                else:
                    pred  = diffusion.sample(lowres, batch_size=lowres.shape[0], t2w=t2w_input)
        else:
            with torch.no_grad():
                pred  = diffusion.sample(lowres, batch_size=lowres.shape[0])
    # else:
    #     lowres    = batch['ADC_condition']
    #     full_size = lowres.shape[-1]
    #     for i in range(batch_size):
    #         lowres[i] = cv2.resize(lowres[i], (full_size//2,full_size//2))
    #         lowres[i] = cv2.resize(lowres[i], (full_size,full_size), interpolation=cv2.INTER_LINEAR)
    #     lowres    = lowres.to(device)
        
    if vae is not None:
        pred   = decode_latent(pred, vae) if diffusion is not None else decode_latent(lowres, vae)
        pred   = pred[:,0,:,:]
        lowres = batch['ADC_condition'].to(device)
        
    # print("orig min/max:", torch.min(lowres),torch.max(lowres))
    # print("rec  min/max:", torch.min(pred),torch.max(pred))
    
    for i in range(batch_size):
        # Plot images
        plot_image(lowres[i],  fig, axes, i, 0)
        plot_image(pred[i],    fig, axes, i, 1)
        plot_image(pred[i],    fig, axes, i, 2, False)
        plot_image(highres[i], fig, axes, i, 3)
        if use_T2W:
            if t2w_image:
                if vae is not None:
                    t2w_input   = decode_latent(t2w_input, vae)[:,0,:,:]
                plot_image(t2w_input[i], fig, axes, i, 4)
            else:
                plot_image(t2w_input[0][i], fig, axes, i, 4)

        plot_error(pred[i], highres[i], fig, axes, i, 2)    

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

 
def visualize_variability(diffusion, dataloader, batch_size, device, use_T2W=False, controlnet=False, output_name="test_image", num_rep=5, avg_std=False):
    ncols = 2+2*num_rep if use_T2W else 2+num_rep
    ncols = ncols+2 if avg_std else ncols
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    
    s = 1 if use_T2W else 0
    if use_T2W:
        axes[0,0].set_title('High res T2W')        
    axes[0,s].set_title('Low res (Input)')
    axes[0,s+1].set_title('High res (Ground truth)')
    for i in range(num_rep):
        axes[0,i+s+2].set_title(f'Super resolution ({i+1})') 
    if avg_std:
        axes[0,ncols-2].set_title('Average Output')
        axes[0,ncols-1].set_title('Std Output')
    
    # Get images 
    batch   = next(iter(dataloader))
    highres = batch['ADC_input'].to(device)
    lowres  = batch['ADC_condition'].to(device)
    
    if vae is not None:
        low_res,_ = encode_latent(lowres, vae)
    
    all_pred = []
    for rep in range(num_rep):
        if use_T2W:
            # Ignoring embedding for injection models
            t2w_input = batch['T2W_condition'].to(device)
            if vae is not None:
                t2w_input,_ = encode_latent(t2w_input, vae)
            with torch.no_grad():
                if controlnet:
                    pred  = diffusion.sample(lowres, batch_size=lowres.shape[0], control=t2w_input)
                else:
                    pred  = diffusion.sample(lowres, batch_size=lowres.shape[0], t2w=t2w_input)
        else:
            with torch.no_grad():
                pred  = diffusion.sample(lowres, batch_size=lowres.shape[0])
        all_pred.append(format_image(pred))
        
    all_pred = np.array(all_pred)
    if avg_std:
        mean_pred = np.mean(all_pred, axis=0)                 
        std_pred  = np.std(all_pred, axis=0)

    for i in range(batch_size):
        # Plot images
        if use_T2W:
            plot_image(t2w_input[i], fig, axes, i, 0)
        plot_image(lowres[i],  fig, axes, i, s)
        plot_image(highres[i], fig, axes, i, s+1)
        for rep in range(num_rep):
            plot_image(all_pred[rep][i], fig, axes, i, rep+s+2)
        if avg_std:
            plot_image(mean_pred[i], fig, axes, i, ncols-2)
            plot_image(std_pred[i],  fig, axes, i, ncols-1, std=True)

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'_variability.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

 
 
def visualize_variability_t2w(diffusion, dataloader, batch_size, device, controlnet=False, output_name="test_image", num_rep=5, avg_std=False):
    ncols = 2+2*num_rep
    ncols = ncols+2 if avg_std else ncols
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    
    axes[0,0].set_title('Low res (Input)')
    axes[0,1].set_title('High res (Ground truth)')
    for i in range(num_rep):
        axes[0,2+i*2].set_title('High res T2W')    
        axes[0,3+i*2].set_title(f'Super resolution ({i+1})') 
    if avg_std:
        axes[0,ncols-2].set_title('Average Output')
        axes[0,ncols-1].set_title('Std Output')
    
    # Get images 
    batch   = next(iter(dataloader))
    highres = batch['ADC_input'].to(device)
    lowres  = batch['ADC_condition'].to(device)
    
    all_pred = []
    all_t2w  = []
    
    t2w_transform = dataloader.dataset.transforms['T2W_condition']  
    for rep in range(num_rep):
        t2w_batch = []
        for p in batch['T2W_path']:
            t2w_img = Image.open(p).convert('L')
            t2w_batch.append(t2w_transform(t2w_img))  
        t2w_input = torch.stack(t2w_batch, dim=0).to(device)
        
        with torch.no_grad():
            if controlnet:
                pred  = diffusion.sample(lowres, batch_size=lowres.shape[0], control=t2w_input)
            else:
                pred  = diffusion.sample(lowres, batch_size=lowres.shape[0], t2w=t2w_input)
                
        all_pred.append(format_image(pred))
        all_t2w.append(format_image(t2w_input))
        
    all_pred = np.array(all_pred)
    all_t2w  = np.array(all_t2w)    
    if avg_std:
        mean_pred = np.mean(all_pred, axis=0)                 
        std_pred  = np.std(all_pred, axis=0)

    for i in range(batch_size):
        # Plot images
        plot_image(lowres[i],  fig, axes, i, 0)
        plot_image(highres[i], fig, axes, i, 1)
        for rep in range(num_rep):
            plot_image(all_t2w[rep][i], fig, axes, i, 2+rep*2)
            plot_image(all_pred[rep][i],fig, axes, i, 3+rep*2)
        if avg_std:
            plot_image(mean_pred[i], fig, axes, i, ncols-2)
            plot_image(std_pred[i],  fig, axes, i, ncols-1, std=True)

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'_variability_t2w.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

 
def plot_image_vae(image, fig, axes, i, j, colorbar=True, std=False):
    try:
        image  = format_image(image)
    except:
        pass
       
    img_plot = axes[i, j].imshow(image,  cmap='gray', vmin=-1, vmax=1)                      
    axes[i, j].axis('off')
    
def visualize_batch_vae(vae, dataloader, accelerator, output_name, greyscale=False):
    batch = next(iter(dataloader))
    x = batch.get("T2W_condition", batch["ADC_input"])
    x = x.to(accelerator.device)
    x = input_to_shape(x, greyscale)
    batch_size = x.size(0)
    
    ncols = 2#5
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    axes[0,0].set_title('Input')
    axes[0,1].set_title('Output 0')
    # axes[0,2].set_title('Output 1')
    # axes[0,3].set_title('Output 2')
    # axes[0,2].set_title('Error')
    
    z, posterior = encode_latent(x, vae)
    x_recon = vae.decode(z)

    for i in range(batch_size):
        plot_image_vae(x[i][0],       fig, axes, i, 0)
        plot_image_vae(x_recon[i][0], fig, axes, i, 1)
        # plot_image(x_recon[i][1], fig, axes, i, 2)
        # plot_image(x_recon[i][2], fig, axes, i, 3)
        # plot_image(x_recon[i][0], fig, axes, i, 2, False)
        # plot_error(x_recon[i][0], x[i][0], fig, axes, i, 2)

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")