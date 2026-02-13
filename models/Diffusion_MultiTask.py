import torch
from torch       import nn
from random      import random
from collections import namedtuple
from functools   import partial

from network_utils      import *
from network_modules    import *
from Diffusion          import Diffusion

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "x_start"])

class Diffusion_MultiTask(Diffusion):
    def __init__(
          self, *args, 
          loss_weights_t2w = {"l1": 1.0, "ssim": 0.5, "perct": 0.0}, 
          noise_rho: float = 0.8, 
          **kwargs
        ):
        super().__init__(*args, **kwargs)    
        
        assert 0.0 <= noise_rho <= 1.0, "noise_rho must be in [0, 1]"
        self.noise_rho = float(noise_rho)

        self.loss_weights_adc = self.loss_weights
        self.loss_weights_t2w = loss_weights_t2w

        print("MultiTask diffusion configured with:")
        print(f"  timestep coupling: shared t")
        print(f"  correlated noise rho={self.noise_rho}")
        print("  ADC loss weights:", self.loss_weights_adc)
        print("  T2W loss weights:", self.loss_weights_t2w)

    def _correlated_noise(self, shape, device, dtype, generator=None):
        """
        Returns (eps_adc, eps_t2w) with correlation rho.
        If generator is provided, sampling is reproducible.
        """
        eps_shared = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        if self.noise_rho == 1.0:
            return eps_shared, eps_shared
        if self.noise_rho == 0.0:
            return eps_shared, torch.randn(shape, device=device, dtype=dtype, generator=generator)

        eps_ind = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        rho = self.noise_rho
        eps_t2w = rho * eps_shared + (1.0 - rho ** 2) ** 0.5 * eps_ind
        return eps_shared, eps_t2w
    
    def _correlated_noise_like(self, x, *, generator=None):
        return self._correlated_noise(x.shape, x.device, x.dtype, generator=generator)

    def _targets_from_objective(self, x0, t, noise, pred):
        """
        For a given task, compute training target based on objective.
        pred is model output (pred_noise / pred_x0 / pred_v depending on objective),
        but we only need target here.
        """
        if self.objective == 'pred_noise':
            return noise
        elif self.objective == 'pred_x0':
            return x0
        elif self.objective == 'pred_v':
            return self.predict_v(x0, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

    def _xstart_from_model_output(self, x_t, t, model_output):
        """
        Convert model output to x_start prediction (in normalized space),
        matching your Diffusion.model_predictions logic.
        """
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        elif self.objective == 'pred_x0':
            x_start = model_output
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x_t, t, v)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        return x_start

    def _randn_like(self, x, *, generator=None):
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

    # ------------------------------------------------------------------------------
    def model_predictions_mt(
        self,
        x_adc, cond_adc,
        x_t2w, cond_t2w,
        t,
        *,
        cond_hbv=None,
        x_self_cond_adc=None, x_self_cond_t2w=None,
        control      = None,
        clip_x_start = False,
        rederive_pred_noise = False,
    ):
        """
        Returns:
            pred_noise_adc, x_start_adc, pred_noise_t2w, x_start_t2w
        using self.objective in {'pred_noise','pred_x0','pred_v'}.
        """
        (out_adc, out_t2w) = self.model(
            x_adc, cond_adc,
            x_t2w, cond_t2w,
            t,
            x_self_cond_adc=x_self_cond_adc,
            x_self_cond_t2w=x_self_cond_t2w,
            control=control,
            cond_hbv=cond_hbv,
        )

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else (lambda z: z)

        def _convert(x, model_output):
            if self.objective == 'pred_noise':
                pred_noise = model_output
                x_start = self.predict_start_from_noise(x, t, pred_noise)
                x_start = maybe_clip(x_start)
                if clip_x_start and rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(x, t, x_start)
                return pred_noise, x_start

            elif self.objective == 'pred_x0':
                x_start = maybe_clip(model_output)
                pred_noise = self.predict_noise_from_start(x, t, x_start)
                return pred_noise, x_start

            elif self.objective == 'pred_v':
                v = model_output
                x_start = self.predict_start_from_v(x, t, v)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(x, t, x_start)
                return pred_noise, x_start

            else:
                raise ValueError(f'unknown objective {self.objective}')

        pred_noise_adc, x_start_adc = _convert(x_adc, out_adc)
        pred_noise_t2w, x_start_t2w = _convert(x_t2w, out_t2w)

        return pred_noise_adc, x_start_adc, pred_noise_t2w, x_start_t2w

    def p_mean_variance_mt(
        self,
        x_adc, cond_adc,
        x_t2w, cond_t2w,
        t,
        *,
        cond_hbv=None,
        x_self_cond_adc=None, x_self_cond_t2w=None,
        control       = None,
        clip_denoised = True
    ):
        """
        Returns:
            model_mean_adc, post_var_adc, post_logvar_adc, x_start_adc,
            model_mean_t2w, post_var_t2w, post_logvar_t2w, x_start_t2w
        """
        pred_noise_adc, x_start_adc, pred_noise_t2w, x_start_t2w = self.model_predictions_mt(
            x_adc, cond_adc, x_t2w, cond_t2w, t,
            x_self_cond_adc=x_self_cond_adc,
            x_self_cond_t2w=x_self_cond_t2w,
            control=control,
            cond_hbv=cond_hbv,
            clip_x_start=clip_denoised,
            rederive_pred_noise=True
        )

        if clip_denoised:
            x_start_adc.clamp_(-1., 1.)
            x_start_t2w.clamp_(-1., 1.)

        mean_adc, var_adc, logvar_adc = self.q_posterior(x_start=x_start_adc, x_t=x_adc, t=t)
        mean_t2w, var_t2w, logvar_t2w = self.q_posterior(x_start=x_start_t2w, x_t=x_t2w, t=t)

        return mean_adc, var_adc, logvar_adc, x_start_adc, mean_t2w, var_t2w, logvar_t2w, x_start_t2w

    @torch.no_grad()
    def p_sample_mt(
        self,
        x_adc, cond_adc,
        x_t2w, cond_t2w,
        t_scalar: int,
        *,
        cond_hbv=None,
        x_self_cond_adc=None, x_self_cond_t2w=None,
        control   = None,
        generator = None
    ):
        """
        One DDPM step at integer timestep t_scalar (shared across both tasks).
        """
        b = x_adc.shape[0]
        device = x_adc.device
        t = torch.full((b,), t_scalar, device=device, dtype=torch.long)

        mean_adc, _, logvar_adc, x0_adc, mean_t2w, _, logvar_t2w, x0_t2w = self.p_mean_variance_mt(
            x_adc, cond_adc, x_t2w, cond_t2w, t,
            x_self_cond_adc=x_self_cond_adc,
            x_self_cond_t2w=x_self_cond_t2w,
            control=control,
            cond_hbv=cond_hbv,
            clip_denoised=True
        )

        if t_scalar > 0:
            noise_adc, noise_t2w = self._correlated_noise_like(x_adc, generator=generator)
        else:
            noise_adc, noise_t2w = 0., 0.

        x_adc_prev = mean_adc + (0.5 * logvar_adc).exp() * noise_adc
        x_t2w_prev = mean_t2w + (0.5 * logvar_t2w).exp() * noise_t2w

        return x_adc_prev, x0_adc, x_t2w_prev, x0_t2w

    @torch.no_grad()
    def p_sample_loop_mt(
        self,
        shape,
        cond_adc, cond_t2w,
        cond_hbv=None,
        *,
        control   = None,
        return_all_timesteps = False,
        generator = None
    ):
        """
        DDPM sampling loop producing (adc, t2w) in [0,1] space.
        shape: (B, C, H, W)
        """
        device = self.device
        x_adc = torch.randn(shape, device=device, generator=generator)
        # correlated init
        _, x_t2w = self._correlated_noise_like(x_adc, generator=generator)

        xs_adc = [x_adc]
        xs_t2w = [x_t2w]
        x0_adc = None
        x0_t2w = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond_adc = x0_adc if self.self_condition else None
            self_cond_t2w = x0_t2w if self.self_condition else None

            x_adc, x0_adc, x_t2w, x0_t2w = self.p_sample_mt(
                x_adc, cond_adc, x_t2w, cond_t2w, t,
                x_self_cond_adc=self_cond_adc,
                x_self_cond_t2w=self_cond_t2w,
                control=control,
                generator=generator,
                cond_hbv=cond_hbv,
            )
            if return_all_timesteps:
                xs_adc.append(x_adc)
                xs_t2w.append(x_t2w)

        if return_all_timesteps:
            x_adc = torch.stack(xs_adc, dim=1)  # [B, T+1, C, H, W]
            x_t2w = torch.stack(xs_t2w, dim=1)

        x_adc = self.unnormalize(x_adc)
        x_t2w = self.unnormalize(x_t2w)
        return x_adc, x_t2w

    @torch.no_grad()
    def ddim_sample_mt(
        self,
        shape,
        cond_adc, cond_t2w,
        *,
        cond_hbv=None,
        control   = None,
        return_all_timesteps = False,
        generator  =None
    ):
        """
        DDIM sampling with optional stochasticity eta. Returns (adc, t2w) in [0,1].
        """
        batch = shape[0]
        device = self.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_adc = torch.randn(shape, device=device, generator=generator)
        # correlated init
        _, x_t2w = self._correlated_noise_like(x_adc, generator=generator)

        xs_adc = [x_adc]
        xs_t2w = [x_t2w]
        x0_adc = None
        x0_t2w = None

        for time, time_next in time_pairs:
            t = torch.full((batch,), time, device=device, dtype=torch.long)

            self_cond_adc = x0_adc if self.self_condition else None
            self_cond_t2w = x0_t2w if self.self_condition else None

            pred_noise_adc, x0_adc, pred_noise_t2w, x0_t2w = self.model_predictions_mt(
                x_adc, cond_adc, x_t2w, cond_t2w, t,
                x_self_cond_adc=self_cond_adc,
                x_self_cond_t2w=self_cond_t2w,
                control=control,
                cond_hbv=cond_hbv,
                clip_x_start=True,
                rederive_pred_noise=True
            )

            if time_next < 0:
                x_adc = x0_adc
                x_t2w = x0_t2w
                if return_all_timesteps:
                    xs_adc.append(x_adc)
                    xs_t2w.append(x_t2w)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # DDIM coefficients
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # correlated noise for the stochastic DDIM term (eta>0)
            if sigma.item() > 0:
                z_adc, z_t2w = self._correlated_noise_like(x_adc, generator=generator)
            else:
                z_adc, z_t2w = 0., 0.

            x_adc = x0_adc * alpha_next.sqrt() + c * pred_noise_adc + sigma * z_adc
            x_t2w = x0_t2w * alpha_next.sqrt() + c * pred_noise_t2w + sigma * z_t2w

            if return_all_timesteps:
                xs_adc.append(x_adc)
                xs_t2w.append(x_t2w)

        if return_all_timesteps:
            x_adc = torch.stack(xs_adc, dim=1)
            x_t2w = torch.stack(xs_t2w, dim=1)

        x_adc = self.unnormalize(x_adc)
        x_t2w = self.unnormalize(x_t2w)
        return x_adc, x_t2w

    @torch.no_grad()
    def sample(self,adc,control=None,batch_size=16,return_all_timesteps=False,t2w=None,hbv=None,perform_uq=False,num_rep=None):
        """
        Returns:
          - if perform_uq: (adc_samples, t2w_samples) each shaped [B, R, C, H, W] (or [B, R, T, C, H, W] if return_all_timesteps)
          - else: (adc, t2w) each shaped [B, C, H, W] (or [B, T, C, H, W] if return_all_timesteps)

        adc and t2w should be the conditioning inputs already on the correct device.
        """
        image_size = self.image_size
        channels = 1  # if yours differ, set from your training setup

        sample_fn = self.p_sample_loop_mt if not self.is_ddim_sampling else self.ddim_sample_mt

        self.model.eval()

        shape = (batch_size, channels, image_size, image_size)

        if self.is_ddim_sampling and perform_uq and num_rep:
            # If user doesn't provide generator, create deterministic per-rep generators.
            # You can also pass a base generator in and advance it; this is simplest.
            seeds = torch.randint(0, 1_000_000, (num_rep,), device=self.device)
            adc_list = []
            t2w_list = []

            for s in seeds:
                g = torch.Generator(device=self.device)
                g.manual_seed(int(s.item()))
                adc_out, t2w_out = sample_fn(
                    shape,
                    adc,
                    t2w,
                    control=control,
                    return_all_timesteps=return_all_timesteps,
                    generator=g,
                    cond_hbv=hbv,
                )
                adc_list.append(adc_out)
                t2w_list.append(t2w_out)

            # stack along rep dimension
            adc = torch.stack(adc_list, dim=1)
            t2w = torch.stack(t2w_list, dim=1)
            return adc, t2w

        else:
            return sample_fn(
                shape,
                adc,
                t2w,
                control=control,
                return_all_timesteps=return_all_timesteps,
                generator=None,
                cond_hbv=hbv
            )

    def p_losses(
        self,
        x0_adc, cond_adc,
        x0_t2w, cond_t2w,
        t,
        *,
        cond_hbv=None,
        defined_target_adc = None, defined_target_t2w = None,
        eval_transform_adc = None, eval_transform_t2w = None,
        control = None,
    ):
        """
        x0_* expected in [0,1] if auto_normalize=True (then we normalize inside)
        """
        b = x0_adc.shape[0]
        assert x0_adc.shape == x0_t2w.shape, "ADC and T2W must have same shape"
        device = x0_adc.device

        # normalize to [-1,1] if enabled
        x0_adc_n = self.normalize(x0_adc)
        x0_t2w_n = self.normalize(x0_t2w)

        # correlated noise
        eps_adc, eps_t2w = self._correlated_noise(x0_adc_n.shape, device=device, dtype=x0_adc_n.dtype)

        # forward diffuse
        x_adc_t = self.q_sample(x_start=x0_adc_n, t=t, noise=eps_adc)
        x_t2w_t = self.q_sample(x_start=x0_t2w_n, t=t, noise=eps_t2w)

        # self-conditioning (optional) — your UNet_Basic ignores it, but we keep parity.
        x_self_cond_adc = None
        x_self_cond_t2w = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # predict x_start for each branch (normalized space), detach
                out_adc_tmp, out_t2w_tmp = self.model(
                    x_adc_t, cond_adc, x_t2w_t, cond_t2w, t,
                    x_self_cond_adc=None, x_self_cond_t2w=None, control=control,
                    cond_hbv=cond_hbv,
                )
                x_self_cond_adc = self._xstart_from_model_output(x_adc_t, t, out_adc_tmp).detach()
                x_self_cond_t2w = self._xstart_from_model_output(x_t2w_t, t, out_t2w_tmp).detach()

        # model outputs
        pred_adc, pred_t2w = self.model(
            x_adc_t, cond_adc, x_t2w_t, cond_t2w, t,
            x_self_cond_adc=x_self_cond_adc, x_self_cond_t2w=x_self_cond_t2w,
            control=control,
            cond_hbv=cond_hbv,
        )

        # x_start preds (normalized)
        x0_adc_pred_n = self._xstart_from_model_output(x_adc_t, t, pred_adc)
        x0_t2w_pred_n = self._xstart_from_model_output(x_t2w_t, t, pred_t2w)

        # targets in normalized space
        target_adc = self._targets_from_objective(x0_adc_n, t, eps_adc, pred_adc)
        target_t2w = self._targets_from_objective(x0_t2w_n, t, eps_t2w, pred_t2w)

        # Loss components
        mse_adc = self.loss.calc_mse(pred_adc, target_adc)
        mse_t2w = self.loss.calc_mse(pred_t2w, target_t2w)

        # For perceptual/ssim, compare in [0,1] space.
        # If you pass defined_target_* you control the “ground truth” used for those metrics.
        pred_adc_01 = self.unnormalize(pred_adc)
        pred_t2w_01 = self.unnormalize(pred_t2w)

        if eval_transform_adc is not None:
            pred_adc_01 = eval_transform_adc(pred_adc_01)
        if eval_transform_t2w is not None:
            pred_t2w_01 = eval_transform_t2w(pred_t2w_01)

        target_adc_01 = self.unnormalize(target_adc) 
        target_t2w_01 = self.unnormalize(target_t2w) 

        perct_adc = self.loss.calc_percept(pred_adc_01, target_adc_01)
        perct_t2w = self.loss.calc_percept(pred_t2w_01, target_t2w_01)

        ssim_adc = self.loss.calc_ssim(pred_adc_01, target_adc_01)
        ssim_t2w = self.loss.calc_ssim(pred_t2w_01, target_t2w_01)

        # Weighted sums (per-sample)
        loss_adc = (
            self.loss_weights_adc.get('mse', 0.0)   * mse_adc +
            self.loss_weights_adc.get('perct', 0.0) * perct_adc +
            self.loss_weights_adc.get('ssim', 0.0)  * ssim_adc
        )
        loss_t2w = (
            self.loss_weights_t2w.get('mse', 0.0)   * mse_t2w +
            self.loss_weights_t2w.get('perct', 0.0) * perct_t2w +
            self.loss_weights_t2w.get('ssim', 0.0)  * ssim_t2w
        )

        # Apply SNR-based reweighting (same t for both => same weights)
        w = extract(self.loss_weight, t, loss_adc.shape)
        loss_adc = loss_adc * w
        loss_t2w = loss_t2w * w

        # Total
        total = (loss_adc + loss_t2w).mean()

        # Return predicted x_start in [0,1] to match your single-task convention
        return (
            self.unnormalize(x0_adc_pred_n),
            self.unnormalize(x0_t2w_pred_n),
            total,
            (mse_adc, perct_adc, ssim_adc),
            (mse_t2w, perct_t2w, ssim_t2w),
            t
        )

    def forward(
        self, 
        x0_adc, cond_adc,
        x0_t2w, cond_t2w,
        *,
        cond_hbv=None,
        defined_target_adc=None, defined_target_t2w=None,
        eval_transform_adc=None, eval_transform_t2w=None,
        control=None,
        **kwargs
    ):
        """
        Joint training forward.
        Expects x0_adc, x0_t2w sizes == image_size.
        """
        b, c, h, w = x0_adc.shape
        assert h == self.image_size and w == self.image_size, f'height/width must be {self.image_size}'
        assert x0_t2w.shape == x0_adc.shape, "ADC and T2W must match shape"

        t = torch.randint(0, self.num_timesteps, (b,), device=x0_adc.device).long()

        return self.p_losses(
            x0_adc, cond_adc, 
            x0_t2w, cond_t2w, 
            t,
            cond_hbv=cond_hbv,
            defined_target_adc=defined_target_adc, defined_target_t2w=defined_target_t2w,
            eval_transform_adc=eval_transform_adc, eval_transform_t2w=eval_transform_t2w,
            control=control,
        )
