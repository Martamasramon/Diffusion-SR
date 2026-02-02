import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('../models')
from Diffusion       import Diffusion
from UNet_Basic      import UNet_Basic 
from UNet_DisC_Diff  import UNet_DisC_Diff
from UNet_DisC_Diff  import UNet_Basic as UNet_Basic_DiscDiff
from UNet_Attn       import UNet_Attn
from load_controlnet import load_pretrained_with_controlnet

import sys
sys.path.append('../')
from dataset         import MyDataset
from trainer_class   import Trainer

folder = '/cluster/project7/backup_masramon/IQT/'
 
def set_device():
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def get_img_size(args, type='basic'):
    if args.upsample:
        img_size = args.img_size*args.down
    elif type=='latent':
        img_size = args.latent_size
    else:
        img_size = args.img_size
    
    return img_size
    
    
def build_UNet(args, type='basic', img_channels=1, discdiff=False):
    img_size = get_img_size(args, type)
    
    if type == 'basic' or type == 'latent':
        print('Building basic UNet model...')
        if discdiff:
            return UNet_Basic_DiscDiff(
                image_size      = img_size,
                hr_condition    = args.use_T2W
            )
        else:
            return UNet_Basic(
                dim             = img_size,
                dim_mults       = tuple(args.dim_mults),
                self_condition  = args.self_condition,
                controlnet      = args.controlnet,
                concat_t2w      = args.use_T2W,
                img_channels    = img_channels
            )
        
    elif type == 'attn':
        print('Building attention UNet model...')
        return UNet_Attn(
        dim             = img_size,
        dim_mults       = tuple(args.dim_mults),
        self_condition  = args.self_condition,
        use_T2W         = args.use_T2W,
        img_channels    = img_channels
    )
    elif type == 'disc_diff':
        print('Building Disc-Diff UNet model...')
        assert args.use_T2W is True
        return UNet_DisC_Diff(
            image_size      = img_size,
            hr_condition    = args.use_T2W
        )
    else:
        raise ValueError(f"Unknown UNet type: {type}")
    
 
def build_diffusion(args, model, type='basic', auto_normalize=True):
    img_size = get_img_size(args, type)

    return Diffusion(
        model,
        image_size          = img_size,
        timesteps           = args.timesteps,
        sampling_timesteps  = args.sampling_timesteps,
        beta_schedule       = args.beta_schedule,
        loss_weights        = {'mse':1, 'ssim':0, 'perct':args.perct_λ},
        auto_normalize      = auto_normalize,
        objective           = args.objective
    )
    
def remap_checkpoints(state, target_model):
    """
    Remap checkpoint keys ONLY if the target model expects the new names.
    
    Args:
        state (dict): checkpoint['model'] state_dict
        target_model (nn.Module): diffusion model you are loading into
    """
    target_keys = set(target_model.state_dict().keys())
    if any('downs.' in k for k in target_keys):
        remapped = {}
        for k, v in state.items():
            nk = k
            nk = nk.replace('downs_label_noise.', 'downs.') 
            nk = nk.replace('final_conv.0.', 'final_conv.')
            remapped[nk] = v    
        return remapped
    else: 
        return state

def load_model(args, model, diffusion, device):
    print('\nLoading checkpoint...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if args.controlnet:
        load_pretrained_with_controlnet(diffusion, checkpoint)
    else:
        missing, unexpected = diffusion.load_state_dict(remap_checkpoints(checkpoint['model'], model), strict=False)
    # print("Missing keys (first 20):",    missing[:20])
    # print("Unexpected keys (first 20):", unexpected[:20])
    
    # Move model to device
    model.eval()
    model.to(device)
    diffusion.model = model
    diffusion.to(device)
    
def load_data(args, data_type='train'):
    print('Loading data...')
    dataset     = MyDataset(
        folder, 
        data_type       = data_type, 
        image_size      = args.img_size, 
        is_finetune     = args.finetune,
        use_mask        = args.use_mask, 
        downsample      = args.down,
        t2w             = args.controlnet | args.use_T2W,
        t2w_offset      = args.t2w_offset, 
        upsample        = args.upsample,
        lowfield        = args.lowfield,
        blank_prob      = args.blank_prob,
    ) 
    if data_type == 'train':
        shuffle = True
    else:
        shuffle = False 
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle) 
    
def build_trainer(args,diffusion,train_dataloader,test_dataloader,accelerator,run,vae=None):
    if args.recon_mse + args.recon_ssim + args.recon_perct > 0:
        print('Using additional image reconstruction loss:')
        image_loss_weights = {'mse': args.recon_mse, 'ssim': args.recon_ssim, 'perct': args.recon_perct}
        print(image_loss_weights)
    else:
        image_loss_weights = None
        
    trainer = Trainer(
        diffusion,
        train_dataloader,
        test_dataloader,
        accelerator,
        use_t2w             = args.controlnet | args.use_T2W,
        finetune_controlnet = args.controlnet,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        train_num_steps     = args.n_epochs,
        gradient_accumulate_every = 2,
        ema_decay           = args.ema_decay,
        amp                 = False,
        results_folder      = args.results_folder,
        save_every          = args.save_every ,
        sample_every        = args.sample_every,
        save_best_and_latest_only = True,
        wandb_run           = run,
        vae                 = vae,
        image_loss_weights  = image_loss_weights
    )
    return trainer

 
