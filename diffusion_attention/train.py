from accelerate import Accelerator

import sys
sys.path.append('../')
from arguments  import args
from init_wandb import get_wandb_obj
from train_test_functions import (
    build_UNet, build_diffusion, load_model,
    set_device, load_data, build_trainer
)

def main():
    assert args.use_T2W 
    accelerator = Accelerator(split_batches=True, mixed_precision='no')

    args.unet_type = 'attn' 
    model     = build_UNet(args)
    diffusion = build_diffusion(args, model)
    
    if args.checkpoint:
        device    = set_device()
        load_model(args, model, diffusion, device)
                
    train_dataloader = load_data(args, 'train')
    test_dataloader  = load_data(args, 'test')

    run = get_wandb_obj(args)
    
    trainer = build_trainer(args,diffusion,train_dataloader,test_dataloader,accelerator,run)
    trainer.train()
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
