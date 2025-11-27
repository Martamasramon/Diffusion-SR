import os
import torch
import torch.nn as nn
from torch.utils.data   import DataLoader
from accelerate         import Accelerator

import sys
sys.path.append('../')
from dataset    import MyDataset       
from arguments  import args          
from init_wandb import get_wandb_obj_VAE

import sys
sys.path.append('../models')
from VAE import build_adc_vae, load_vae, train_step, val_step

def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')

    folder = '/cluster/project7/backup_masramon/IQT/'
    train_dataset = MyDataset(
        folder,
        data_type       = 'train',
        image_size      = args.img_size,    
        use_mask        = args.use_mask,
        downsample      = args.down,
        t2w             = args.use_T2W,        
        t2w_offset      = args.t2w_offset,
        upsample        = args.upsample,
    )
    val_dataset = MyDataset(
        folder,
        data_type       = 'test',
        image_size      = args.img_size,
        use_mask        = args.use_mask,
        downsample      = args.down,
        t2w             = args.use_T2W,        
        t2w_offset      = args.t2w_offset,
        upsample        = args.upsample,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,  batch_size=args.batch_size, shuffle=False,  num_workers=4, pin_memory=True)

    # Log in WanDB
    run = get_wandb_obj_VAE(args)
    
    # Build VAE
    greyscale = True
    vae = load_vae(args.vae_type, greyscale)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode        = 'min',    # reduce when loss stops decreasing
        factor      = 0.5,      
        patience    = 5,        
        threshold   = 1e-4,     # ignore tiny fluctuations
        min_lr      = 1e-7,      
        verbose     = True      
    )
    vae, optimizer, train_loader, val_loader = accelerator.prepare(vae, optimizer, train_loader, val_loader)

    n_epochs = args.n_epochs
    save_dir = args.results_folder
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = 1000000000
    for epoch in range(1, n_epochs+1):
        print(f"[Epoch {epoch}/{n_epochs}] ")
        train_loss, train_rec, train_kl = train_step(vae, train_loader, accelerator, optimizer, greyscale)
        val_loss,   val_rec,   val_kl   = val_step  (vae, val_loader, accelerator, greyscale )
        
        run.log({
            "train_total":  train_loss,
            "train_recon":  train_rec,
            "train_kl":     train_kl,
            "val_total":    val_loss,
            "val_recon":    val_rec,
            "val_kl":       val_kl,
            "lr":           optimizer.param_groups[0]['lr']      
        })
        
        scheduler.step(val_loss)
        
        if accelerator.is_main_process:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(save_dir, f"vae_best.ckpt")
                torch.save({"state_dict": accelerator.unwrap_model(vae).state_dict()}, ckpt_path)
                print(f"Saved VAE checkpoint to {ckpt_path}")
                
            if epoch % args.save_every == 0:
                ckpt_path = os.path.join(save_dir, f"vae_{epoch}.ckpt")
                torch.save({"state_dict": accelerator.unwrap_model(vae).state_dict()}, ckpt_path)
                print(f"Saved VAE checkpoint to {ckpt_path}")
                


if __name__ == "__main__":
    print("Training ADC AutoencoderKL...")
    
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
