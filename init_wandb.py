import wandb
from datetime import datetime

def get_wandb_obj(args):
    # year_month_day_hour_minute
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M")

    run = wandb.init(
        entity  =   "marta-masramon",
        project =   "ADC-SR",
        name    =   f"{args.results_folder[2:]}_{date_str}",
        config={
            "name":             args.results_folder,
            "image_size":       args.img_size,
            "downsampling":     args.down, 
            "timesteps":        args.timesteps, 
            "sampling_timesteps":args.sampling_timesteps, 
            "beta_schedule":     args.beta_schedule, 
            "batch_size":       args.batch_size, 
            "initial_lr":       args.lr,
            "epochs":           args.n_epochs,
            "ema_decay":        args.ema_decay, 
            "blank_prob":       args.blank_prob, 
            "t2w_offset":       args.t2w_offset, 
            "save_every":       args.save_every, 
            "sample_every":     args.sample_every, 
            "use_T2W":          args.use_T2W, 
            "use_mask":         args.use_mask, 
            "finetune":         args.finetune,             
            "controlnet":       args.controlnet, 
            "upsample":         args.upsample, 
        },
    )

    return run

def get_wandb_obj_VAE(args):
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    
    run = wandb.init(
        entity  =   "marta-masramon",
        project =   "VAE",
        name    =   f"{args.results_folder[2:]}_{date_str}",
        config={
            "name":             args.results_folder,
            "image_size":       args.img_size,
            "downsampling":     args.down, 
            "initial_lr":       args.lr,
            "epochs":           args.n_epochs,
            "blank_prob":       args.blank_prob, 
            "t2w_offset":       args.t2w_offset, 
            "use_T2W":          args.use_T2W, 
            "use_mask":         args.use_mask, 
            "finetune":         args.finetune,    
            "vae_type":         args.vae_type,     
        },
    )

    return run