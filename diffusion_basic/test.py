import time
import torch
import argparse
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.append('../models')
from network_utils  import *
from Diffusion      import Diffusion
from UNet_Basic     import UNet_Basic

import sys
sys.path.append('../')
from dataset         import MyDataset
from test_functions  import *
from arguments       import args

folder = '/cluster/project7/backup_masramon/IQT/'
 
def main():
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet_Basic(
        dim             = args.img_size,
        dim_mults       = tuple(args.dim_mults),
        self_condition  = args.self_condition,
        controlnet      = args.controlnet,
        concat_t2w      = args.use_T2W,
    )

    diffusion = Diffusion(
        model,
        image_size          = args.img_size,
        timesteps           = args.timesteps,
        sampling_timesteps  = args.sampling_timesteps,
        beta_schedule       = args.beta_schedule,
    )

    print('Loading checkpoint...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    diffusion.load_state_dict(checkpoint['model'])
    
    # Move model to device
    model.eval()
    model.to(device)
    diffusion.model = model
    diffusion.to(device)
    
    print('Loading data...')
    dataset     = MyDataset(
        folder, 
        data_type       = 'val', 
        image_size      = args.img_size, 
        is_finetune     = args.finetune,
        use_mask        = args.use_mask, 
        downsample      = args.down,
        t2w             = args.controlnet | args.use_T2W,
        t2w_offset      = args.t2w_offset, 
    ) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print('Visualising...')
    save_name = args.save_name if args.save_name is not None else os.path.basename(os.path.dirname(args.checkpoint))
    test_data = 'HistoMRI' if args.finetune else 'PICAI'
    
    # visualize_variability_t2w(diffusion, dataloader, args.batch_size, device, controlnet=args.controlnet, output_name=f'{save_name}_{test_data}')
    visualize_variability(diffusion, dataloader, args.batch_size, device, controlnet=args.controlnet, output_name=f'{save_name}_{test_data}', use_T2W=args.use_T2W)
    # visualize_batch(diffusion, dataloader, args.batch_size, device, controlnet=args.controlnet, output_name=f'{save_name}_{test_data}', use_T2W=args.use_T2W)
    
    # print('Evaluating...')
    # evaluate_results(diffusion, dataloader, device, args.batch_size, use_T2W=args.use_T2W, controlnet=args.controlnet)
    

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
