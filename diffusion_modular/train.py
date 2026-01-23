import numpy as np
import os
import time
import torch
torch.cuda.set_device(0)
import glob
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

import sys
sys.path.append('../models')
from Diffusion_Modular  import Diffusion_Modular
from UNet_Modular       import UNet_Modular

import sys
sys.path.append('../')
from trainer_class  import Trainer_mod
from dataset        import MyDataset_lowField
from arguments      import args

folder = '/cluster/project7/backup_masramon/IQT/'

def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')

    model = UNet_Modular(
        in_modalities       = {"adc_noisy": 1, "adc_lr": 1, "t2w_lr": 1},   
        out_modalities      = {"adc": 1, "t2w": 1},
        dim                 = args.img_size,
        dim_mults           = tuple(args.dim_mults),
        default_fusion      = "film_gated",
        fusion_by_output    = {"adc": "film_gated", "t2w": "xattn"},
        modality_drop_prob  = args.modality_drop_prob,      
        never_drop          = ["adc_lr"],                   
    )
    
    diffusion = Diffusion_Modular(
        model,
        image_size          = args.img_size,
        timesteps           = args.timesteps,
        sampling_timesteps  = args.sampling_timesteps,
        beta_schedule       = args.beta_schedule,
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        diffusion.load_state_dict(checkpoint['model'], strict=False)
                
    # Dataset and dataloader   
    train_dataset = MyDataset_lowField(
        folder, 
        data_type       = 'train', 
        image_size      = args.img_size, 
    ) 
    test_dataset = MyDataset_lowField(
        folder, 
        data_type       = 'test', 
        image_size      = args.img_size
    ) 
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size = args.batch_size, shuffle=False)
        
    trainer = Trainer_mod(
        diffusion,
        train_dataloader,
        test_dataloader,
        accelerator,
        use_t2w_embed       = True,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        train_num_steps     = args.n_epochs,
        gradient_accumulate_every = 2,
        ema_decay           = args.ema_decay,
        amp                 = False,
        results_folder      = args.results_folder,
        save_every          = args.save_every ,
        sample_every        = args.sample_every,
        save_best_and_latest_only = True
    )

    trainer.train()
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
