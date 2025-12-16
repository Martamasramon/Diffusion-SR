import os
import torch
import torch.nn as nn
from torch.utils.data   import DataLoader
from accelerate         import Accelerator

import sys
sys.path.append('../')
from dataset    import MyDataset       
from arguments  import args          
from test_functions  import *

import sys
sys.path.append('../models')
from VAE import build_adc_vae, load_vae, val_step

 
def main():
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder = '/cluster/project7/backup_masramon/IQT/'
    test_dataset = MyDataset(
        folder,
        data_type       = 'val',
        image_size      = args.img_size,    
        use_mask        = args.use_mask,
        downsample      = args.down,
        t2w             = args.use_T2W,        
        t2w_offset      = args.t2w_offset,
        upsample        = args.upsample,
    )
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build VAE
    vae = load_vae(args.vae_type, args.greyscale, args.checkpoint)
    vae.to(device)

    # Visualise & evaluate results
    print('Visualizing...') 
    visualize_batch(None, test_loader, args.batch_size, device, output_name=args.save_name, vae=vae)
    
    # print('Evaluating...')    
    # _,_,_ = val_step (vae, test_loader, accelerator, args.greyscale )

    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
