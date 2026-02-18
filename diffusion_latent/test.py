import os
import sys
sys.path.append('../')
from test_functions import visualize_batch, evaluate_results
from arguments      import args
from train_test_functions import (
    build_UNet, build_diffusion, load_model,
    set_device, load_data
)
sys.path.append('../models')
from models.VAE import load_vae

def main():
    device    = set_device()
    
    args.unet_type = 'latent' 
    model     = build_UNet(args, img_channels=3)
    diffusion = build_diffusion(args, model, auto_normalize=False)
    load_model(args, model, diffusion, device)
    
    vae = load_vae(args.vae_type, args.greyscale)
    vae.to(device)
    vae.eval()
    
    dataloader = load_data(args, 'val')

    save_name = args.save_name if args.save_name is not None else os.path.basename(os.path.dirname(args.checkpoint))
    test_data = 'HistoMRI' if args.finetune else 'PICAI'
    
    print('Visualising...')
    perform_uq = args.perform_uq
    num_rep    = args.num_repeats if perform_uq else None
    visualize_batch(
        args,
        diffusion, 
        dataloader, 
        device, 
        output_name = f'{save_name}_{test_data}'
    )
    
    print('Evaluating...')
    evaluate_results(
        args, 
        diffusion, 
        dataloader, 
        device, 
    )
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
