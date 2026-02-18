from accelerate import Accelerator
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
from dataset        import MyDataset3D
from arguments      import args
from init_wandb import get_wandb_obj
from train_test_functions import (
    build_UNet, build_diffusion, load_model,
    set_device, load_data, build_trainer
)

folder = '/cluster/project7/backup_masramon/PI-CAI/'

def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')
    
    model     = build_UNet(args, type='basic')
    diffusion = build_diffusion(args, model)
    
    if args.checkpoint:
        device    = set_device()
        load_model(args, model, diffusion, device)
        
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

    run = get_wandb_obj(args)

    trainer = build_trainer(args,diffusion,train_dataloader,test_dataloader,accelerator,run)
    trainer.train()
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
