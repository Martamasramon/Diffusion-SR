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
from Diffusion      import Diffusion
from UNet_3D        import UNet_3D

import sys
sys.path.append('../')
from trainer_class  import Trainer
from dataset        import MyDataset3D
from arguments      import args

folder = '/cluster/project7/backup_masramon/PI-CAI/'

def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')

    model = UNet_3D(
        dim             = args.img_size,
        dim_mults       = tuple(args.dim_mults),
        self_condition  = args.self_condition,
        controlnet      = args.controlnet,
        concat_t2w      = args.use_T2W
    )
    
    diffusion = Diffusion(
        model,
        image_size          = args.img_size,
        timesteps           = args.timesteps,
        sampling_timesteps  = args.sampling_timesteps,
        beta_schedule       = args.beta_schedule,
        perct_λ             = args.perct_λ
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        diffusion.load_state_dict(checkpoint['model'])
        
    # Dataset and dataloader
    train_dataset = MyDataset3D(
        folder, 
        data_type       = 'train', 
        blank_prob      = args.blank_prob,
        image_size      = args.img_size, 
        is_finetune     = args.finetune, 
        use_mask        = args.use_mask, 
        downsample      = args.down,
        t2w             = args.controlnet | args.use_T2W
    ) 
    test_dataset = MyDataset3D(
        folder, 
        data_type       = 'test', 
        image_size      = args.img_size, 
        is_finetune     = args.finetune, 
        use_mask        = args.use_mask, 
        downsample      = args.down,
        t2w             = args.controlnet | args.use_T2W
    ) 
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size = args.batch_size, shuffle=False)

    trainer = Trainer(
        diffusion,
        train_dataloader,
        test_dataloader,
        accelerator,
        use_t2w             = args.controlnet | args.use_T2W,
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
