import lpips
import torch
import torch.nn.functional as F
from torch          import nn
from torch.amp      import autocast
from einops         import rearrange, reduce
from tqdm.auto      import tqdm
from random         import random
from functools      import partial
from collections    import namedtuple
from piq            import ssim
from loss           import VGGPerceptualLoss

from network_utils   import *
from network_modules import *

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'x_start'])

class Losses(nn.Module):
    def __init__(self):
        super().__init__()
        self.perct_loss = VGGPerceptualLoss()
        
    def calc_mse(self, x_pred, x_target):
        loss = F.mse_loss(x_pred, x_target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        return loss
    
    def calc_percept(self, x_pred, x_target):
        predicted = x_pred.clamp(0.0, 1.0)
        target    = x_target.clamp(0.0, 1.0)
        loss = self.perct_loss(predicted, target)
        return loss
    
    def calc_ssim(self, x_pred, x_target):
        predicted = x_pred.clamp(0.0, 1.0)
        target    = x_target.clamp(0.0, 1.0)
        val       = ssim(predicted, target, data_range=1.0)
        return 1.0 - val
        
    
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps               = 1000,
        sampling_timesteps      = None,
        objective               = 'pred_v',  ## vs 'pred_noise'
        beta_schedule           = 'sigmoid', ## vs 'linear'
        schedule_fn_kwargs      = dict(),
        ddim_sampling_eta       = 0.,        ## vs 1.
        auto_normalize          = True,      # False for latent diffusion !!!
        min_snr_gamma           = 5,
        loss_weights            = {'mse':1, 'ssim':0, 'perct':0},
    ):
        super().__init__()

        self.model                  = model
        self.input_img_channels     = self.model.input_img_channels
        self.mask_channels          = self.model.mask_channels
        self.self_condition         = self.model.self_condition
        self.image_size             = image_size
        self.objective              = objective
        
        self.controlnet = self.model.controlnet 
        self.concat_t2w = self.model.concat_t2w
                
        self.channels = self.model.input_img_channels

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas              = 1. - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps,          = betas.shape
        self.num_timesteps  = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling   = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta  = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas',                betas)
        register_buffer('alphas_cumprod',       alphas_cumprod)
        register_buffer('alphas_cumprod_prev',  alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod',              torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',    torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',     torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',      torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped',   torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1',             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2',             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight https://arxiv.org/abs/2303.09556
        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            register_buffer('loss_weight', torch.ones_like(snr))
        elif objective == 'pred_x0':
            register_buffer('loss_weight', snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize      = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize    = unnormalize_to_zero_to_one  if auto_normalize else identity
        
        self.loss           = Losses()
        self.loss_weights   = loss_weights
        
        print("Loss weights:", self.loss_weights)

    @property
    def device(self):
        return next(self.parameters()).device


    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise )


    def predict_noise_from_start(self, x_t, t, x0):
        return ((   extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))


    def predict_v(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start )


    def predict_start_from_v(self, x_t, t, v):
        return (extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)


    def q_posterior(self, x_start, x_t, t):
        posterior_mean                  = ( extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                                            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance              = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped  = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def model_predictions(self, x, low_res, t, x_self_cond = None, control=None, t2w=None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model.forward(x, low_res, t, x_self_cond, control=control, t2w=t2w)
        maybe_clip  = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    

    def p_mean_variance(self, x, low_res, t, x_self_cond = None, control=None, t2w=None, clip_denoised = True):
        preds = self.model_predictions(x, low_res, t, x_self_cond, control=control, t2w=t2w)
        x_start = preds.x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start


    @torch.no_grad()
    def p_sample(self, x, low_res, t, x_self_cond=None, control=None, t2w=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, low_res, batched_times, x_self_cond, clip_denoised=True, control=control, t2w=t2w)
        noise       = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img    = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start


    @torch.no_grad()
    def p_sample_loop(self, shape, low_res, control=None, return_all_timesteps = False, t2w=None):
        batch, device = shape[0], self.device
        img           = torch.randn(shape, device = device)
        imgs          = [img]
        x_start       = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, low_res, t, x_self_cond=self_cond, control=control, t2w=t2w)
            imgs.append(img)

        img = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        img = self.unnormalize(img)
        return img


    @torch.no_grad()
    def ddim_sample(self, shape, low_res, control=None, return_all_timesteps = False, t2w=None, generator=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img     = torch.randn(shape, device = device, generator=generator)
        imgs    = [img]
        x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            preds   = self.model_predictions(img, low_res, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, control=control, t2w=t2w)
            x_start = preds.x_start
            
            if time_next < 0:
                img = preds.x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn(img.shape,device=img.device,dtype=img.dtype,generator=generator)

            img = preds.x_start * alpha_next.sqrt() + \
                  c * preds.pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, low_res, control=None, batch_size = 16, return_all_timesteps = False, t2w=None, perform_uq=False, num_rep=None):
        image_size, channels = self.image_size, self.channels
        sample_fn            = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        self.model.eval()

        if self.is_ddim_sampling and perform_uq and num_rep:

            seeds = torch.randint(0, 1e6, (num_rep,), device=self.device)
            imgs_pred = []

            for seed in seeds:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed.item())
                # torch.cuda.manual_seed(seed.item())
                img_pred = sample_fn((batch_size, channels, image_size, image_size), low_res, control=control, return_all_timesteps = return_all_timesteps, t2w=t2w, generator=generator)
                imgs_pred.append(img_pred)
            
            imgs_pred = torch.stack(imgs_pred, dim=1)  # Shape: (batch_size, num_reruns, C, H, W)
            # imgs_pred_mean = imgs_pred.mean(dim=1)
            # imgs_pred_std  = imgs_pred.std(dim=1)

            return imgs_pred # Return all samples for uncertainty quantification

        else:
            return sample_fn((batch_size, channels, image_size, image_size), low_res, control=control, return_all_timesteps = return_all_timesteps, t2w=t2w)


    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, t2w=None, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        
    def p_losses(self, 
                x_start, 
                low_res, 
                t, 
                control, 
                t2w, 
                defined_target,             
                eval_transform,
                noise   = None, 
        ):
        b, c, h, w              = x_start.shape
        noise                   = default(noise, lambda: torch.randn_like(x_start))
           
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # self-conditioning. 50% of the time, predict x_start from current set of times and condition with unet
        #                   slows down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x,low_res, t, control=control, t2w=t2w).x_start  
                x_self_cond.detach_()

        # predict and take gradient step
        prediction   = self.model(x, low_res, t, x_self_cond, control=control, t2w=t2w)
        x_start_pred = self.model_predictions(x, low_res, t, x_self_cond=x_self_cond, control=control, t2w=t2w).x_start   # predicted clean latent

        if self.objective == 'pred_noise':
            target = noise  
        elif self.objective == 'pred_x0':
            target = x_start  
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')
                    
        norm_pred   = self.unnormalize(prediction) if eval_transform is None else eval_transform(self.unnormalize(prediction)) 
        norm_target = self.unnormalize(target)    if eval_transform is None else defined_target                            
            
        # mse loss before normalising, perceptual and ssim after normalising
        mse_loss    = self.loss.calc_mse(prediction, target)
        perct_loss  = self.loss.calc_percept(norm_pred, norm_target)
        ssim_loss   = self.loss.calc_ssim(norm_pred, norm_target)

        loss = (self.loss_weights['mse']   * mse_loss   + 
                self.loss_weights['perct'] * perct_loss + 
                self.loss_weights['ssim']  * ssim_loss )
        loss *= extract(self.loss_weight, t, loss.shape)
        # Maybe try loss = mse_loss.mean()

        return self.unnormalize(x_start_pred), loss.mean(), mse_loss, perct_loss, ssim_loss, t


    def forward(self, 
                input_img, 
                condition_adc, 
                condition_t2w  = None, 
                control        = None, 
                defined_target = None,             
                eval_transform = None,
                *args, 
                **kwargs
        ):
        b, c, h, w, device, img_size, = *input_img.shape, input_img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        return self.p_losses(self.normalize(input_img), condition_adc, t, control, condition_t2w, defined_target, eval_transform, *args, **kwargs)
