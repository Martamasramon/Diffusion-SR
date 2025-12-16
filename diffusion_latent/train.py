import numpy as np
import os
import time
import torch
import glob
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

import sys
sys.path.append('../models')
from Diffusion        import Diffusion
from UNet_Basic       import UNet_Basic
from load_controlnet  import load_pretrained_with_controlnet
from VAE              import load_vae

import sys
sys.path.append('../')
from trainer_class  import Trainer
from dataset        import MyDataset
from arguments      import args
from init_wandb     import get_wandb_obj

folder = '/cluster/project7/backup_masramon/IQT/'

def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')
    
    model = UNet_Basic(
        dim             = args.latent_size,
        dim_mults       = tuple(args.dim_mults),
        self_condition  = args.self_condition,
        controlnet      = args.controlnet,
        concat_t2w      = args.use_T2W,
        img_channels    = 3
    )
    
    diffusion = Diffusion(
        model,
        image_size          = args.latent_size,
        timesteps           = args.timesteps,
        sampling_timesteps  = args.sampling_timesteps,
        beta_schedule       = args.beta_schedule,
        loss_weights        = {'mse':1, 'ssim':0, 'perct':0},
        auto_normalize      = False,
        objective           = args.objective
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        if args.controlnet:
            load_pretrained_with_controlnet(diffusion, checkpoint)
        else:
            diffusion.load_state_dict(checkpoint['model'])
    
    vae = load_vae(args.vae_type, args.greyscale)

    # Dataset and dataloader
    train_dataset = MyDataset(
        folder, 
        data_type       = 'train', 
        blank_prob      = args.blank_prob,
        image_size      = args.img_size, 
        is_finetune     = args.finetune, 
        use_mask        = args.use_mask, 
        downsample      = args.down,
        t2w             = args.controlnet | args.use_T2W,
        t2w_offset      = args.t2w_offset,
        upsample        = args.upsample,
    ) 
    test_dataset = MyDataset(
        folder, 
        data_type       = 'test', 
        image_size      = args.img_size, 
        is_finetune     = args.finetune, 
        use_mask        = args.use_mask, 
        downsample      = args.down,
        t2w             = args.controlnet | args.use_T2W,
        t2w_offset      = args.t2w_offset,
        upsample        = args.upsample,
    ) 
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size = args.batch_size, shuffle=False)

    if args.recon_mse + args.recon_ssim + args.recon_perct > 0:
        print('Using additional image reconstruction loss:')
        image_loss_weights = {'mse': args.recon_mse, 'ssim': args.recon_ssim, 'perct': args.recon_perct}
        print(image_loss_weights)
    else:
        image_loss_weights = None
     
    run = get_wandb_obj(args)       
    trainer = Trainer(
        diffusion,
        train_dataloader,
        test_dataloader,
        accelerator,
        use_t2w             = args.controlnet | args.use_T2W,
        finetune_controlnet = args.controlnet,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        train_num_steps     = args.n_epochs,
        gradient_accumulate_every = 2,
        ema_decay           = args.ema_decay,
        amp                 = False,
        results_folder      = args.results_folder,
        save_every          = args.save_every ,
        sample_every        = args.sample_every,
        save_best_and_latest_only = True,
        wandb_run           = run,
        vae                 = vae,
        image_loss_weights  = image_loss_weights
    )

    trainer.train()
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
