import math
from pathlib import Path

import torch
from torch.optim import Adam
from torchvision import transforms as T, utils
from tqdm.auto   import tqdm
from ema_pytorch import EMA

from models.network_utils import (
    cycle, num_to_groups, has_int_squareroot, divisible_by, extract, exists
)
from models.VAE import encode_latent, decode_latent
from models.Diffusion   import Losses
from transforms import downsample_transform

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_dataloader,
        test_dataloader,
        accelerator,
        *,
        use_T2W_embed               = False,
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
        val_every                   = 100,
        num_samples                 = 25,
        results_folder              = './results',
        amp                         = False,
        mixed_precision_type        = 'fp16',
        split_batches               = True,
        inception_block_idx         = 2048,
        max_grad_norm               = 1.,
        save_best_and_latest_only   = True,
        wandb_run                   = None, 
        vae                         = None,
        image_loss_weights          = {'mse':1, 'ssim':0, 'perct': 0.01},
    ):
        super().__init__()

        self.accelerator    = accelerator
        self.model          = diffusion_model
        self.channels       = diffusion_model.input_img_channels
        self.is_ddim_sampling    = diffusion_model.is_ddim_sampling

        self.img_size       = img_size        
        self.use_T2W        = self.model.use_T2W 
        self.use_T2W_embed  = use_T2W_embed
        self.use_HBV        = self.model.use_HBV 
        self.finetune_controlnet = finetune_controlnet
        
        assert save_every%val_every==0 and sample_every%val_every==0, "save_every and sample_every should be multiples of val_every for consistent milestone logging and validation frequency."
        
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
        self.val_every      = val_every

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
                t2w_in  = data['T2W_condition'] if (self.use_T2W and self.model.controlnet is None) else None
                hbv_in  = data['HBV'] if self.use_HBV else None
                if 'ADC_target' in data.keys():
                    defined_target = data['ADC_target'] 
                    eval_transform = downsample_transform(self.img_size) 
                else:
                    defined_target, eval_transform = None, None
                
                losses = {}

                if self.use_T2W_embed:
                    data['T2W_embed'] = [t.squeeze(1) for t in data['T2W_embed']]
                    prediction, loss, losses['mse'], losses['perct'], losses['ssim'], t = self.model(data['ADC_input'], data['ADC_condition'], data['T2W_embed'], hbv_in, control, defined_target, eval_transform)
                else:
                    prediction, loss, losses['mse'], losses['perct'], losses['ssim'], t = self.model(data['ADC_input'], data['ADC_condition'], t2w_in           , hbv_in, control, defined_target, eval_transform)
                
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
            sample_t2w      = data['T2W_condition'][:self.num_samples].to(self.accelerator.device) if (self.model.controlnet is not None) or self.model.use_T2W else None
            sample_hbv      = data['HBV'][:self.num_samples].to(self.accelerator.device) if self.model.use_HBV else None
            
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
                hbv     = sample_hbv[start:end] if sample_hbv is not None else None 

                if low_res.shape[0] == 0:
                    break  # no more valid conditioning inputs
                
                if 'T2W_embed' in data:
                    t2w_embed = sample_t2w_embed[start:end]
                    images    = self.ema.ema_model.sample(batch_size=low_res.shape[0], low_res=low_res, t2w=t2w_embed, hbv=hbv)
                else:
                    control = t2w if self.model.controlnet else None
                    t2w_in  = t2w if (self.model.use_T2W and self.model.controlnet is None) else None
                    images  = self.ema.ema_model.sample(batch_size=low_res.shape[0], low_res=low_res, control=control, t2w=t2w_in, hbv=hbv)
                    
                all_images_list.append(images)
                start = end
        
        all_images = torch.cat(all_images_list, dim = 0)
        
        if self.vae is not None:
            all_images = decode_latent(all_images, self.vae)[:, 0, :, :].unsqueeze(1)
        
        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

    
    def _fmt_losses_for_pbar(self, losses: dict) -> str:
        if not losses:
            return ""

        # single-task
        if all(k in losses for k in ("mse", "perct", "ssim")):
            return f"(MSE {losses['mse']:.4f}, perct {losses['perct']:.4f}, SSIM {losses['ssim']:.4f})"

        # multitask (ADC / T2W)
        parts = []
        for p in ("adc", "t2w"):
            keys = [f"{p}_mse", f"{p}_perct", f"{p}_ssim"]
            if any(k in losses for k in keys):
                vals = [f"{k.split('_')[1]} {losses[k]:.4f}" for k in keys if k in losses]
                parts.append(f"{p.upper()}: " + ", ".join(vals))
        if parts:
            return "(" + " | ".join(parts) + ")"

        # fallback: show first few numeric entries
        return "(" + ", ".join(
            f"{k} {v:.4f}" for k, v in list(losses.items())[:4] if isinstance(v, (int, float))
        ) + ")"

                
    def train(self):
        accelerator = self.accelerator

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                # Calculate loss
                _, total_loss_train, total_losses_train = self.calc_loss(train=True)
                
                # Validate every few steps to reduce overhead 
                do_val = (self.step % self.val_every == 0)
                if do_val:
                    with torch.no_grad():
                        data, total_loss_val, total_losses_val = self.calc_loss(train=False)
                else:
                    total_loss_val, total_losses_val = None, None
                
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
                            if 'mse' in total_losses_val:
                                val_mse = total_losses_val['mse']
                            else:
                                val_mse = total_losses_val.get('adc_mse', total_loss_val)
                            if self.best_mse > val_mse:
                                self.best_mse = val_mse
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)
                
                # Log losses to wandb
                if self.run is not None:
                    log_losses = {'Train - total': total_loss_train}
                    for k, v in total_losses_train.items():
                        log_losses['Train - ' + k] = v
                        
                    if do_val:
                        log_losses['Val - total'] = total_loss_val
                        for k, v in total_losses_val.items():
                            log_losses['Val - ' + k] = v

                    self.run.log(log_losses)
                    
                # Update pbar
                if self.step % 100 == 0:
                    desc = f"Train loss: {total_loss_train:.4f} {self._fmt_losses_for_pbar(total_losses_train)}"
                    if do_val:
                        desc += f"\nValidation loss: {total_loss_val:.4f} {self._fmt_losses_for_pbar(total_losses_val)}"

                    pbar.set_description(desc)
                    pbar.update(100)

        accelerator.print('training complete')



class Trainer_MultiTask(Trainer):
    """
    Multi-task trainer for a Diffusion_MultiTask model that outputs ADC SR + T2W denoising jointly.

    Expected model.forward(...) output (recommended):
        pred_adc, pred_t2w, total_loss, logs_dict, t

    If your Diffusion_MultiTask currently returns:
        pred_adc, pred_t2w, total_loss, (mse_adc, perct_adc, ssim_adc), (mse_t2w, perct_t2w, ssim_t2w), t
    that's also supported (logs_dict will be constructed).

    Batch keys (defaults):
        ADC_input, ADC_condition, (optional ADC_target)
        T2W_condition (used as clean T2W image), (optional T2W_target)
    """

    def __init__(
        self,
        *args,
        save_t2w_samples: bool = True,
        modality_dropout_p_adc: float = 0.0,
        modality_dropout_p_t2w: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.save_t2w_samples = save_t2w_samples
        
        self.modality_dropout_p_adc = modality_dropout_p_adc
        self.modality_dropout_p_t2w = modality_dropout_p_t2w

        if self.vae is not None:
            self.image_loss = Losses()
            self.image_loss = self.accelerator.prepare(self.image_loss)

    def _move_batch_to_device(self, data: dict):
        for key, value in data.items():
            try:
                data[key] = value.to(self.accelerator.device)
            except:
                data[key] = [i.to(self.accelerator.device) for i in value]
        return data

    def _maybe_encode_latents(self, data: dict):
        if self.vae is None:
            return data

        keys_to_encode = ["ADC_input", "ADC_condition", "ADC_target", "T2W_condition", "T2W_input", "T2W_target"]
        for k in keys_to_encode:
            if k in data and isinstance(data[k], torch.Tensor):
                data[k], _ = encode_latent(data[k], self.vae)
        return data
    
    def _parse_multitask_forward_output(self, out):
        """
        Accepts either:
          (pred_adc, pred_t2w, total, logs_dict, t)
        or
          (pred_adc, pred_t2w, total, (mse_adc, perct_adc, ssim_adc), (mse_t2w, perct_t2w, ssim_t2w), t)

        Returns:
          pred_adc, pred_t2w, total_loss, logs_dict, t
        """
        if not isinstance(out, (tuple, list)):
            raise TypeError(f"Model output must be tuple/list, got {type(out)}")

        if len(out) == 5 and isinstance(out[3], dict):
            pred_adc, pred_t2w, total_loss, logs, t = out
            return pred_adc, pred_t2w, total_loss, logs, t

        if len(out) == 6:
            pred_adc, pred_t2w, total_loss, adc_losses, t2w_losses, t = out
            mse_adc, perct_adc, ssim_adc = adc_losses
            mse_t2w, perct_t2w, ssim_t2w = t2w_losses
            logs = {
                "adc_mse":   mse_adc.mean()   if isinstance(mse_adc,   torch.Tensor) else mse_adc,
                "adc_perct": perct_adc.mean() if isinstance(perct_adc, torch.Tensor) else perct_adc,
                "adc_ssim":  ssim_adc.mean()  if isinstance(ssim_adc,  torch.Tensor) else ssim_adc,
                "t2w_mse":   mse_t2w.mean()   if isinstance(mse_t2w,   torch.Tensor) else mse_t2w,
                "t2w_perct": perct_t2w.mean() if isinstance(perct_t2w, torch.Tensor) else perct_t2w,
                "t2w_ssim":  ssim_t2w.mean()  if isinstance(ssim_t2w,  torch.Tensor) else ssim_t2w,
            }
            return pred_adc, pred_t2w, total_loss, logs, t

        raise ValueError(f"Unrecognized multitask model output signature with len={len(out)}")

    def _add_recon_image_loss(
        self,
        *,
        pred_latent: torch.Tensor,
        target_img: torch.Tensor,
        t: torch.Tensor,
        loss: torch.Tensor,
        logs: dict,
        prefix: str,
    ):
        """
        Adds decoded-image reconstruction loss to `loss` and fills `logs` with:
        {prefix}_recon_mse, {prefix}_recon_perct, {prefix}_recon_ssim, {prefix}_recon_total

        Assumes:
        - pred_latent is what your diffusion predicts (latent if VAE exists, else image)
        - target_img is the image-space target (already on device, matching expected range)
        - self.image_loss is Losses() with calc_mse/calc_percept/calc_ssim
        - self.image_loss_weights is dict with mse/perct/ssim
        - uses diffusion SNR weighting via extract(self.model.loss_weight, t, per_sample_loss.shape)
        """
        if self.image_loss_weights is None or self.vae is None:
            return loss, logs

        pred_img = decode_latent(pred_latent, self.vae)
        # your decode returns [B, C, H, W]; keep first channel if needed
        if pred_img.ndim == 4 and pred_img.shape[1] != 1:
            pred_img = pred_img[:, 0:1]
        if target_img.ndim == 4 and target_img.shape[1] != 1:
            target_img = target_img[:, 0:1]

        # Compute per-sample components
        recon_mse   = self.image_loss.calc_mse(pred_img, target_img)          # [B]
        recon_perct = self.image_loss.calc_percept(pred_img, target_img)      # usually scalar or [B]
        recon_ssim  = self.image_loss.calc_ssim(pred_img, target_img)         # [B] if implemented like yours

        # Make sure perct is per-sample for weighting; if it's scalar, expand
        if isinstance(recon_perct, torch.Tensor) and recon_perct.ndim == 0:
            recon_perct = recon_perct.expand_as(recon_mse)

        # Weighted per-sample total
        recon_total = (
            self.image_loss_weights.get("mse", 0.0)   * recon_mse +
            self.image_loss_weights.get("perct", 0.0) * recon_perct +
            self.image_loss_weights.get("ssim", 0.0)  * recon_ssim
        )

        # SNR-based weighting (same as your diffusion loss)
        recon_total = recon_total * extract(self.model.loss_weight, t, recon_total.shape)

        # Add to main loss
        loss = loss + recon_total.mean()

        # Logging (means)
        logs[f"{prefix}_recon_mse"]   = recon_mse.mean()
        logs[f"{prefix}_recon_perct"] = recon_perct.mean() if isinstance(recon_perct, torch.Tensor) else float(recon_perct)
        logs[f"{prefix}_recon_ssim"]  = recon_ssim.mean()
        logs[f"{prefix}_recon_total"] = recon_total.mean()

        return loss, logs
    
    def _apply_modality_dropout(self, data: dict):
        """
        During training, randomly zero out ADC_condition and/or T2W_condition.

        - If exclusive: at most one modality is dropped per sample.
        - Operates on whatever is in data (latent or image space), so call AFTER _maybe_encode_latents.
        """
        p_adc = float(self.modality_dropout_p_adc or 0.0)
        p_t2w = float(self.modality_dropout_p_t2w or 0.0)
        if p_adc <= 0.0 and p_t2w <= 0.0:
            return data

        device = self.accelerator.device

        cond_adc = data.get("ADC_condition", None)
        cond_t2w = data.get("T2W_condition", None)
        if not isinstance(cond_adc, torch.Tensor) or not isinstance(cond_t2w, torch.Tensor):
            return data
        B = cond_adc.shape[0]

        # per-sample categorical: {none, drop_adc, drop_t2w} -> p = [1 - p_adc - p_t2w, p_adc, p_t2w]
        p_none = max(0.0, 1.0 - (p_adc + p_t2w))
        probs = torch.tensor([p_none, p_adc, p_t2w], device=device, dtype=torch.float32)
        probs = probs / probs.sum().clamp_min(1e-8)

        # sample categories
        cat = torch.multinomial(probs, num_samples=B, replacement=True)  # 0,1,2
        drop_adc = (cat == 1)
        drop_t2w = (cat == 2)

        # broadcast masks to [B, C, H, W] (or any [B, ...])
        while drop_adc.ndim < cond_adc.ndim:
            drop_adc = drop_adc.view(*drop_adc.shape, 1)
        while drop_t2w.ndim < cond_t2w.ndim:
            drop_t2w = drop_t2w.view(*drop_t2w.shape, 1)

        # zero-fill (blank condition)
        data["ADC_condition"] = torch.where(drop_adc, torch.zeros_like(cond_adc), cond_adc)
        data["T2W_condition"] = torch.where(drop_t2w, torch.zeros_like(cond_t2w), cond_t2w)

        return data

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
        total_losses = {}  # dynamic log keys

        if self.vae is not None:
            total_losses["recon_mse"] = 0.0
            total_losses["recon_perct"] = 0.0
            total_losses["recon_ssim"] = 0.0

        dataloader = self.train_dataloader if train else self.test_dataloader

        for _ in range(self.gradient_accumulate_every):
            data = next(dataloader)

            # Move to device, encode latents (if needed) & apply modality dropout 
            data = self._move_batch_to_device(data)
            data = self._maybe_encode_latents(data)
            data = self._apply_modality_dropout(data)

            with self.accelerator.autocast():
                hbv_in = data.get("HBV_condition", None)
                out = self.model(
                    data["ADC_input"],
                    data["ADC_condition"],
                    data["T2W_input"],
                    data["T2W_condition"],
                    cond_hbv=hbv_in,
                )
                
                
                pred_adc, pred_t2w, loss, logs, t = self._parse_multitask_forward_output(out)

                # Optional: latent recon loss for ADC prediction (same as your original logic)
                if self.vae is not None and self.image_loss_weights is not None:
                    loss, logs = self._add_recon_image_loss(
                        pred_latent = pred_adc,
                        target_img  = data["ADC_input"],   
                        t           = t,
                        loss        = loss,
                        logs        = logs,
                        prefix      = "adc",
                    )
                    loss, logs = self._add_recon_image_loss(
                        pred_latent = pred_t2w,
                        target_img  = data["T2W_input"],  
                        t           = t,
                        loss        = loss,
                        logs        = logs,
                        prefix      = "t2w",
                    )

                # accumulate
                total_loss += loss.detach().item() / self.gradient_accumulate_every
                for k, v in logs.items():
                    if isinstance(v, torch.Tensor):
                        v = v.mean().detach().item()
                    else:
                        v = float(v)
                    total_losses[k] = total_losses.get(k, 0.0) + v / self.gradient_accumulate_every

            if train:
                self.accelerator.backward(loss)

        return data, total_loss, total_losses

    def sample_images(self, data):
        """
        Save grids of ADC and (optionally) T2W samples from EMA model.
        """
        with torch.no_grad():
            milestone = self.step // self.sample_every
            batches = num_to_groups(self.num_samples, self.batch_size)

            cond_adc_full = data["ADC_condition"][:self.num_samples].to(self.accelerator.device)
            cond_t2w_full = data["T2W_condition"][:self.num_samples].to(self.accelerator.device)

            all_adc = []
            all_t2w = []

            start = 0
            total = cond_adc_full.shape[0]

            for n in batches:
                end = min(start + n, total)
                cond_adc = cond_adc_full[start:end]
                cond_t2w = cond_t2w_full[start:end]

                if cond_adc.shape[0] == 0:
                    break

                # EMA model returns (adc, t2w) for multitask sample()
                hbv_in = data["HBV_condition"][start:end] if self.model.use_HBV else None
                adc_s, t2w_s = self.ema.ema_model.sample(
                    adc      = cond_adc,
                    t2w      = cond_t2w,
                    cond_hbv = hbv_in,
                    batch_size=cond_adc.shape[0],
                    return_all_timesteps=False
                )

                all_adc.append(adc_s)
                all_t2w.append(t2w_s)

                start = end

            all_adc = torch.cat(all_adc, dim=0)
            all_t2w = torch.cat(all_t2w, dim=0)

            # decode latents if needed
            if self.vae is not None:
                all_adc = decode_latent(all_adc, self.vae)[:, 0, :, :].unsqueeze(1)
                all_t2w = decode_latent(all_t2w, self.vae)[:, 0, :, :].unsqueeze(1)

            nrow = int(math.sqrt(self.num_samples))
            utils.save_image(all_adc, str(self.results_folder / f"sample-adc-{milestone}.png"), nrow=nrow)

            if self.save_t2w_samples:
                utils.save_image(all_t2w, str(self.results_folder / f"sample-t2w-{milestone}.png"), nrow=nrow)
