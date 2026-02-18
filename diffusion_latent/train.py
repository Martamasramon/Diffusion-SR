from accelerate import Accelerator

import sys
sys.path.append('../')
from arguments  import args
from init_wandb import get_wandb_obj
from train_test_functions import (
    build_UNet, build_diffusion, load_model,
    set_device, load_data, build_trainer
)
sys.path.append('../models')
from models.VAE import load_vae

def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')
    
    args.unet_type == 'latent'
    model     = build_UNet(args, img_channels=3)
    diffusion = build_diffusion(args, model, auto_normalize=False)
    
    if args.checkpoint:
        device    = set_device()
        load_model(args, model, diffusion, device)
    
    vae = load_vae(args.vae_type, args.greyscale)

    train_dataloader = load_data(args, 'train')
    test_dataloader  = load_data(args, 'test')
     
    run = get_wandb_obj(args)    
    
    trainer = build_trainer(args,diffusion,train_dataloader,test_dataloader,accelerator,run, vae)
    trainer.train()
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
