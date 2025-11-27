import torch
from ldm.models.autoencoder import AutoencoderKL 
from torch.nn.functional import mse_loss
import yaml

import sys
sys.path.append('../')
from test_functions  import plot_image, format_image, plot_error

def read_config(folder):
    with open(f"/cluster/project7/ProsRegNet_CellCount/CriDiff/autoencoder/pretrained/{folder}/config.yaml", 'r') as file:
        config = yaml.safe_load(file)
        
    return config['model']['params']

def build_adc_vae(folder):
    config   = read_config(folder)
    ddconfig = config['ddconfig']

    vae = AutoencoderKL(
        ddconfig   = ddconfig,
        embed_dim  = ddconfig["z_channels"],
        monitor    = "val/rec_loss",
        lossconfig = config["lossconfig"]
    )
    return vae

def input_to_shape(x, greyscale):
    x = x * 2.0 - 1.0  # dataset gives [0,1]; AutoencoderKL expects [-1,1]
    if not greyscale: 
        x = x.repeat(1, 3, 1, 1) # Expects RGB
    return x

def load_vae(ckpt_folder, greyscale=False, ckpt_path=None):
    vae  = build_adc_vae(ckpt_folder)
    
    if ckpt_path is None:
        ckpt = torch.load(f"./pretrained/{ckpt_folder}/model.ckpt", map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(f"./{ckpt_path}.ckpt", map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    
    if greyscale:
        sd = adapt_to_gray(vae, sd)
    
    vae.load_state_dict(sd)
    return vae
    
def adapt_to_gray(vae, sd):
    sd_new = vae.state_dict()

    for k, w_new in sd_new.items():
        if k not in sd:
            continue

        w_pre = sd[k]

        # First conv: (out_ch, in_ch, k, k)
        if w_pre.ndim == 4 and w_pre.shape[1] == 3 and w_new.shape[1] == 1:
            sd_new[k] = w_pre.mean(dim=1, keepdim=True)

        # Last conv: (out_ch=3, in_ch, k, k) -> (1, in_ch, k, k)
        elif w_pre.ndim == 4 and w_pre.shape[0] == 3 and w_new.shape[0] == 1:
            sd_new[k] = w_pre.mean(dim=0, keepdim=False).unsqueeze(0)

        # Same shape: copy directly
        elif w_pre.shape == w_new.shape:
            sd_new[k] = w_pre

    return sd_new

def train_step(vae, train_loader, accelerator, optimizer, greyscale=False, kl_weight = 1e-6):
    vae.train()
    train_rec = 0.0
    train_kl  = 0.0
    total_loss= 0.0
    n_train   = 0

    for batch in train_loader:
        x = batch.get("T2W_condition", batch["ADC_input"])  # (B, 1, H, W) from your transforms
        x = x.to(accelerator.device)
        x = input_to_shape(x, greyscale)   
            
        posterior = vae.encode(x)
        z = posterior.sample()
        x_recon = vae.decode(z)

        rec_loss = mse_loss(x_recon, x)
        kl_loss  = posterior.kl().mean() # posterior.kl() returns per-pixel; mean over batch

        loss = rec_loss + kl_weight * kl_loss

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()

        bs = x.size(0)
        train_rec += rec_loss.detach().item() * bs
        train_kl  += kl_loss.detach().item() * bs
        total_loss+= loss.detach().item() * bs
        n_train   += bs

    train_rec /= n_train
    train_kl  /= n_train
    total_loss /= n_train

    print(f"Train loss: {total_loss:.4e} (rec={train_rec:.4e}, kl={train_kl:.4e}) ")
    return total_loss, train_rec, train_kl

def val_step(vae, val_loader, accelerator, greyscale=False, kl_weight = 1e-6):
    vae.eval()
    val_rec   = 0.0
    val_kl    = 0.0
    total_loss = 0.0
    n_test     = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch.get("T2W_condition", batch["ADC_input"])
            x = x.to(accelerator.device)
            x = input_to_shape(x, greyscale)
            
            posterior = vae.encode(x)
            z = posterior.sample()
            x_recon = vae.decode(z)

            rec_loss = mse_loss(x_recon, x)
            kl_loss  = posterior.kl().mean()
            loss     = rec_loss + kl_weight * kl_loss
            
            bs = x.size(0)
            val_rec += rec_loss.item() * bs
            val_kl  += kl_loss.item() * bs
            total_loss+= loss.detach().item() * bs
            n_test   += bs

    val_rec /= n_test
    val_kl  /= n_test
    total_loss /= n_test

    print(f"Test loss: {total_loss:.4e} (rec={val_rec:.4e}, kl={val_kl:.4e}) ")
    return total_loss, val_rec, val_kl

    
def visualize_batch(vae, dataloader, accelerator, output_name, greyscale=False):
    batch = next(iter(dataloader))
    x = batch.get("T2W_condition", batch["ADC_input"])
    x = x.to(accelerator.device)
    x = input_to_shape(x, greyscale)
    batch_size = x.size(0)
    
    ncols = 3
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    axes[0,0].set_title('Input')
    axes[0,1].set_title('Output')
    axes[0,2].set_title('Error')
    
    with torch.no_grad():
        posterior = vae.encode(x)
        z = posterior.sample()
        x_recon = vae.decode(z)

    for i in range(batch_size):
        plot_image(x[i],       fig, axes, i, 0)
        plot_image(x_recon[i], fig, axes, i, 1)
        plot_image(x_recon[i], fig, axes, i, 2, False)
        plot_error(x_recon, x, fig, axes, i, 2)

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")