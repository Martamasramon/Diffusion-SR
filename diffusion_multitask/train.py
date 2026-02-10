from accelerate import Accelerator

import sys
sys.path.append('../')
from arguments  import args
from init_wandb import get_wandb_obj
from train_test_functions import (
    build_UNet, build_diffusion, load_model,
    set_device, load_data, build_trainer
)

import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def main():
    args.use_T2W    = True
    args.unet_type  = 'multitask'
    
    accelerator  = Accelerator(split_batches=True, mixed_precision='fp16')
    
    import os
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.current_device =", torch.cuda.current_device())
    print("device name =", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("accelerator device =", accelerator.device)
    
    model     = build_UNet(args)
    diffusion = build_diffusion(args, model)
    
    if args.checkpoint:
        device    = set_device()
        load_model(args, model, diffusion, device)
    
    train_dataloader = load_data(args, 'train')
    test_dataloader  = load_data(args, 'test')

    run = get_wandb_obj(args)
    # run = None
    
    trainer = build_trainer(args,diffusion,train_dataloader,test_dataloader,accelerator,run)
    trainer.train()
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
