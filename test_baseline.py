import time
import torch
import argparse
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.append('../models')
from models.network_utils import *

import sys
sys.path.append('../')
from dataset         import MyDataset
from test_functions  import *

folder = '/cluster/project7/backup_masramon/IQT/'


parser = argparse.ArgumentParser("Diffusion")

parser.add_argument('--img_size',           type=int,  default=64)
parser.add_argument('--save_name',          type=str,  default='test_image')
parser.add_argument('--batch_size',         type=int,  default=15)

parser.add_argument('--finetune',           action='store_true')
parser.add_argument('--use_mask',           action='store_true')
parser.set_defaults(use_mask  = False)
parser.set_defaults(finetune = False)
args, unparsed = parser.parse_known_args()
 
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = 'HistoMRI' if args.finetune else 'PICAI'
    
    print('Loading data...')
    dataset     = MyDataset(
        folder + data_folder, 
        args.img_size, 
        data_type       = 'val', 
        is_finetune     = args.finetune,
        use_mask        = args.use_mask,
    ) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print('Visualising...')
    visualize_batch(None, dataloader, args.batch_size, device, output_name=f'{args.save_name}_{data_folder}')
    
    print('Evaluating...')
    evaluate_results(None, dataloader, device, args.batch_size)
    

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
