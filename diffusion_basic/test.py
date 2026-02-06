import sys
sys.path.append('../')
from test_functions import *
from arguments      import args
from train_test_functions import (
    build_UNet, build_diffusion, load_model,
    set_device, load_data
)
 
def main():
    device    = set_device()
    
    model     = build_UNet(args)
    diffusion = build_diffusion(args, model)
    load_model(args, model, diffusion, device)
    
    dataloader = load_data(args, 'val')

    print('Visualising...')
    save_name = args.save_name if args.save_name is not None else os.path.basename(os.path.dirname(args.checkpoint))
    test_data = 'HistoMRI' if args.finetune else 'PICAI'
    
    # visualize_variability_t2w(diffusion, dataloader, args.batch_size, device, controlnet=args.controlnet, output_name=f'{save_name}_{test_data}')
    # visualize_variability(diffusion, dataloader, args.batch_size, device, controlnet=args.controlnet, output_name=f'{save_name}_{test_data}', use_T2W=args.use_T2W)
    visualize_batch(diffusion, dataloader, args.batch_size, device, controlnet=args.controlnet, output_name=f'{save_name}_{test_data}', use_T2W=args.use_T2W)
    
    # print('Evaluating...')
    evaluate_results(diffusion, dataloader, device, args.batch_size, use_T2W=args.use_T2W, controlnet=args.controlnet)
    

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
