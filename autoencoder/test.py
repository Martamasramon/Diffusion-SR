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
from VAE import build_adc_vae, load_vae, visualize_batch, val_step

 
def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')

    folder = '/cluster/project7/backup_masramon/IQT/'
    test_dataset = MyDataset(
        folder,
        data_type       = 'train',
        image_size      = args.img_size,    
        use_mask        = args.use_mask,
        downsample      = args.down,
        t2w             = args.use_T2W,        
        t2w_offset      = args.t2w_offset,
        upsample        = args.upsample,
    )
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build VAE
    greyscale = True
    vae = load_vae(args.vae_type, greyscale, args.checkpoint)
    
    # Visualise & evaluate results
    print('Visualizing...') 
    output_name = args.checkpoint
    visualize_batch(vae, test_loader, accelerator, output_name, greyscale=False)
    
    print('Evaluating...')    
    _,_,_ = val_step (vae, test_loader, accelerator, greyscale )

    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
