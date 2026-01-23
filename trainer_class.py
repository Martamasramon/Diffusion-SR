import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.functional as F

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from tqdm.auto import tqdm

import sys
sys.path.append('/models')
from network_utils   import *
from network_modules import *
from VAE import encode_latent, decode_latent
from Diffusion   import Losses
from ema_pytorch import EMA
from transforms import downsample_transform


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_dataloader,
        test_dataloader,
        accelerator,
        *,
        use_t2w                     = False,
        use_t2w_embed               = False,
        finetune_controlnet         = False,
        batch_size                  = 16,
        img_size                    = 64,
        gradient_accumulate_every   = 1,
        lr                          = 1e-4,
        train_num_steps             = 100000,
        ema_update_every            = 10,
        ema_decay                   = 0.995,
        adam_betas                  = (0.9, 0.99),
        save_every                  = 5000,
        sample_every                = 2000,
        num_samples                 = 25,
        results_folder              = './results',
        amp                         = False,
        mixed_precision_type        = 'fp16',
        split_batches               = True,
        inception_block_idx         = 2048,
        max_grad_norm               = 1.,
        save_best_and_latest_only   = False,
        wandb_run                   = None, 
        vae                         = None,
        image_loss_weights          = {'mse':1, 'ssim':0, 'perct': 0.01},
    ):
        super().__init__()

        self.accelerator    = accelerator
        self.model          = diffusion_model
        self.channels       = diffusion_model.input_img_channels
        is_ddim_sampling    = diffusion_model.is_ddim_sampling

        self.img_size       = img_size        
        self.use_t2w        = use_t2w
        self.use_t2w_embed  = use_t2w_embed
        self.finetune_controlnet = finetune_controlnet
        
        if self.finetune_controlnet:
            assert getattr(self.model, "controlnet", None) is not None, "finetune_controlnet_only=True but model.controlnet is None."

            # Freeze main model only
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.controlnet.parameters():
                p.requires_grad = True

            self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            self.trainable_params = list(self.model.parameters())
        
        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples    = num_samples
        self.save_every     = save_every
        self.sample_every   = sample_every

        self.batch_size                = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps    = train_num_steps
        self.image_size         = diffusion_model.image_size
        self.max_grad_norm      = max_grad_norm

        self.train_dataloader = cycle(self.accelerator.prepare(train_dataloader))
        self.test_dataloader  = cycle(self.accelerator.prepare(test_dataloader))

        self.opt = Adam(self.trainable_params, lr = lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # from apex import amp
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if save_best_and_latest_only:
            self.best_mse = 10000

        self.save_best_and_latest_only = save_best_and_latest_only
        self.run = wandb_run 
        
        # Prep VAE if latent diffusion
        self.vae = accelerator.prepare(vae) if vae is not None else None
        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False
            self.image_loss = Losses()
            self.image_loss = self.accelerator.prepare(self.image_loss)
                
        self.image_loss_weights  = image_loss_weights

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step':     self.step,
            'model':    self.accelerator.get_state_dict(self.model),
            'opt':      self.opt.state_dict(),
            'ema':      self.ema.state_dict(),
            'scaler':   self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def calc_loss(self, train=True):
        if train:
            if self.finetune_controlnet:
                self.model.eval()
                self.model.controlnet.train()
            else:
                self.model.train()
        else:
            self.model.eval()
            
        total_loss   = 0
        total_losses = {'mse': 0.0, 'perct':0.0, 'ssim':0.0}
        if self.vae is not None:
            total_losses['recon_mse'], total_losses['recon_perct'], total_losses['recon_ssim'] = 0.0, 0.0, 0.0
        
        dataloader = self.train_dataloader if train else self.test_dataloader

        for _ in range(self.gradient_accumulate_every):
            data = next(dataloader)
            
            if self.vae is not None:
                data['Image_target'] = data['ADC_condition']
                
            for key, value in data.items():
                try:
                    data[key] = value.to(self.accelerator.device)
                except:
                    data[key] = [i.to(self.accelerator.device) for i in value]
                
                if self.vae is not None and key in ['ADC_input','ADC_condition','ADC_target','T2W_condition']:
                    data[key],_ = encode_latent(data[key], self.vae)
                     
            with self.accelerator.autocast():
                control = data['T2W_condition'] if self.model.controlnet else None
                t2w_in  = data['T2W_condition'] if (self.use_t2w and self.model.controlnet is None) else None
                if 'ADC_target' in data.keys():
                    defined_target = data['ADC_target'] 
                    eval_transform = downsample_transform(self.img_size) 
                else:
                    defined_target, eval_transform = None, None
                
                losses = {}
                if self.use_t2w_embed:
                    data['T2W_embed'] = [t.squeeze(1) for t in data['T2W_embed']]
                    prediction, loss, losses['mse'], losses['perct'], losses['ssim'], t = self.model(data['ADC_input'], data['ADC_condition'], data['T2W_embed'], control, defined_target, eval_transform)
                else:
                    prediction, loss, losses['mse'], losses['perct'], losses['ssim'], t = self.model(data['ADC_input'], data['ADC_condition'], t2w_in,            control,  defined_target, eval_transform)
                
                if self.vae is not None and self.image_loss_weights is not None:
                    reconstruction = decode_latent(prediction, self.vae)[:, 0, :, :].unsqueeze(1) 
                    img_target     = data['Image_target']                                         

                    recon_mse      = self.image_loss.calc_mse(reconstruction, img_target)       
                    recon_perct    = self.image_loss.calc_percept(reconstruction, img_target) 
                    recon_ssim     = self.image_loss.calc_ssim(reconstruction, img_target)     

                    losses['recon_mse']   = recon_mse.mean()
                    losses['recon_perct'] = recon_perct
                    losses['recon_ssim']  = recon_ssim

                    # build per-sample recon loss for SNR weighting
                    recon_loss = ( self.image_loss_weights['mse']   * recon_mse +
                                self.image_loss_weights['perct'] * recon_perct +
                                self.image_loss_weights['ssim']  * recon_ssim )   
                    recon_loss *= extract(self.model.loss_weight, t, recon_loss.shape)          
                    loss = loss + recon_loss.mean()
                    
                total_loss += loss.detach().item() / self.gradient_accumulate_every                        
                for name, val in losses.items():
                    if isinstance(val, torch.Tensor) and val.ndim > 0:
                        val = val.mean()
                    total_losses[name] += val.detach().item() / self.gradient_accumulate_every
                    
            if train:
                self.accelerator.backward(loss)
                    
        return data, total_loss, total_losses
                    
    def sample_images(self, data):
        with torch.no_grad():
            milestone       = self.step // self.sample_every
            batches         = num_to_groups(self.num_samples, self.batch_size)
            sample_lowres   = data['ADC_condition'][:self.num_samples].to(self.accelerator.device)
            sample_t2w      = data['T2W_condition'][:self.num_samples].to(self.accelerator.device) if (self.model.controlnet is not None) or self.model.concat_t2w else None
            
            if 'T2W_embed' in data:
                sample_t2w_embed = []
                for i in range(4):
                    sample_t2w_embed.append(data['T2W_embed'][i][:self.num_samples].to(self.accelerator.device))
            
            # all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, cond=sample_lowres), batches))
            all_images_list = []
            start = 0
            total = sample_lowres.shape[0]

            for n in batches:
                end = min(start + n, total)
                low_res = sample_lowres[start:end]
                t2w     = sample_t2w[start:end] if sample_t2w is not None else None

                if low_res.shape[0] == 0:
                    break  # no more valid conditioning inputs
                
                if 'T2W_embed' in data:
                    t2w_embed = sample_t2w_embed[start:end]
                    images    = self.ema.ema_model.sample(batch_size=low_res.shape[0], low_res=low_res, t2w=t2w_embed)
                else:
                    control = t2w if self.model.controlnet else None
                    t2w_in  = t2w if (self.model.concat_t2w and self.model.controlnet is None) else None
                    images  = self.ema.ema_model.sample(batch_size=low_res.shape[0], low_res=low_res, control=control, t2w=t2w_in)
                    
                all_images_list.append(images)
                start = end
        
        all_images = torch.cat(all_images_list, dim = 0)
        
        if self.vae is not None:
            all_images = decode_latent(all_images, self.vae)[:, 0, :, :].unsqueeze(1)
        
        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                
    def train(self):
        accelerator = self.accelerator

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                # Calculate loss
                _,    total_loss_train, total_losses_train = self.calc_loss(train=True)
                data, total_loss_val , total_losses_val    = self.calc_loss(train=False)
                
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.trainable_params, self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # Sample images
                    if self.step != 0 and divisible_by(self.step, self.sample_every):
                        self.ema.ema_model.eval()
                        self.sample_images(data)
                        
                    # Save model 
                    if self.step != 0 and divisible_by(self.step, self.save_every):
                        milestone = self.step // self.save_every
                        if self.save_best_and_latest_only:
                            val_mse = total_losses_val.get('mse', total_loss_val)
                            if self.best_mse > val_mse:
                                self.best_mse = val_mse
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)
                
                if self.run is not None:
                    log_losses = {
                        'Train - total': total_loss_train,
                        'Val - total':   total_loss_val,
                    }
                    for key in total_losses_train:
                        log_losses['Train - '+key] = total_losses_train[key]
                        log_losses['Val - '+key]   = total_losses_val[key]
                    self.run.log(log_losses)
                    
                # Update pbar
                if self.step % 100 == 0:
                    pbar.set_description(f"Train loss: {total_loss_train:.4f} (MSE: {total_losses_train['mse']:.4f},  perct: {total_losses_train['perct']:.4f}, SSIM: {total_losses_train['ssim']:.4f},)\n"+
                                         f"Test loss:  {total_loss_val:.4f} (MSE: {total_losses_val['mse']:.4f},  perct: {total_losses_val['perct']:.4f}, SSIM: {total_losses_val['ssim']:.4f},)")
                    pbar.update(100)

        accelerator.print('training complete')



class Trainer_mod(Trainer):
    def __init__(self, *args, **kwargs):
        # IMPORTANT: forward args to base Trainer
        super().__init__(*args, **kwargs)

    def calc_loss(self, train=True):
        if train:
            if self.finetune_controlnet:
                self.model.eval()
                self.model.controlnet.train()
            else:
                self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_losses = {}  # dynamic keys (mse/perct/ssim/loss_t2w/...)

        if self.vae is not None:
            # keep your latent recon logging if you still want it
            total_losses["recon_mse"] = 0.0
            total_losses["recon_perct"] = 0.0
            total_losses["recon_ssim"] = 0.0

        dataloader = self.train_dataloader if train else self.test_dataloader

        for _ in range(self.gradient_accumulate_every):
            data = next(dataloader)

            # move to device
            if self.vae is not None:
                data["Image_target"] = data["ADC_condition"]

            for key, value in data.items():
                try:
                    data[key] = value.to(self.accelerator.device)
                except:
                    data[key] = [i.to(self.accelerator.device) for i in value]

                # latent encode if needed
                if self.vae is not None and key in ["ADC_input", "ADC_condition", "ADC_target", "T2W_condition", "T2W_target"]:
                    data[key], _ = encode_latent(data[key], self.vae)

            with self.accelerator.autocast():
                control = data["T2W_condition"] if getattr(self.model, "controlnet", None) else None

                # ADC eval target
                if "ADC_target" in data:
                    defined_target = data["ADC_target"]
                    eval_transform = downsample_transform(self.img_size)
                else:
                    defined_target, eval_transform = None, None

                # NEW: T2W condition + optional clean target
                t2w_def = data["T2W_condition"] if self.use_t2w else None
                t2w_tgt = data.get("T2W_target", None)  # requires dataset to provide it for supervised T2W branch

                # IMPORTANT: Diffusion_mod returns (prediction, loss, logs, t)
                prediction, loss, logs, t = self.model(
                    data["ADC_input"],
                    data["ADC_condition"],
                    t2w_def,
                    control,
                    t2w_target=t2w_tgt,
                    defined_target=defined_target,
                    eval_transform=eval_transform,
                )

                # optional latent recon loss (same as your original logic)
                if self.vae is not None and self.image_loss_weights is not None:
                    reconstruction = decode_latent(prediction, self.vae)[:, 0, :, :].unsqueeze(1)
                    img_target = data["Image_target"]

                    recon_mse = self.image_loss.calc_mse(reconstruction, img_target)
                    recon_perct = self.image_loss.calc_percept(reconstruction, img_target)
                    recon_ssim = self.image_loss.calc_ssim(reconstruction, img_target)

                    # per-sample recon loss for SNR weighting
                    recon_loss = (
                        self.image_loss_weights["mse"] * recon_mse
                        + self.image_loss_weights["perct"] * recon_perct
                        + self.image_loss_weights["ssim"] * recon_ssim
                    )
                    recon_loss *= extract(self.model.loss_weight, t, recon_loss.shape)
                    loss = loss + recon_loss.mean()

                    # log recon components
                    logs["recon_mse"] = recon_mse.mean()
                    logs["recon_perct"] = recon_perct.mean() if isinstance(recon_perct, torch.Tensor) else torch.tensor(0.0, device=self.device)
                    logs["recon_ssim"] = recon_ssim.mean()

                # accumulate totals
                total_loss += loss.detach().item() / self.gradient_accumulate_every

                for name, val in logs.items():
                    if isinstance(val, torch.Tensor):
                        val = val.mean()
                        val_item = val.detach().item()
                    else:
                        val_item = float(val)

                    total_losses[name] = total_losses.get(name, 0.0) + val_item / self.gradient_accumulate_every

            if train:
                self.accelerator.backward(loss)

        return data, total_loss, total_losses

    # You can keep sample_images identical for now (it will still sample ADC via diffusion.sample()).
    # If later you extend sampling to output T2W too, we can adjust here.
