import sys
sys.path.append('../')
from test_functions import *
from arguments      import args
from train_test_functions import (
    build_UNet, build_diffusion, load_model,
    set_device, load_data
)
sys.path.append('../models')
from VAE import load_vae

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
    visualize_batch(diffusion, dataloader, args.batch_size, device, controlnet=args.controlnet, output_name=f'{save_name}_{test_data}', use_T2W=args.use_T2W, vae=vae, perform_uq=perform_uq, num_rep=num_rep)
    
    # print('Evaluating...')
    evaluate_results(diffusion, dataloader, device, args.batch_size, use_T2W=args.use_T2W, controlnet=args.controlnet)
    

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
