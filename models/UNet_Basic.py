import torch
from torch      import nn
from functools  import partial

from network_utils   import exists, default
from network_modules import (
    ResnetBlock, Downsample, Upsample,
    Residual, PreNorm, LinearCrossAttention,
    make_time_mlp, 
)
from ControlNet import ControlNet

class UNet_Basic(nn.Module):
    def __init__(
        self,
        dim:            int   = 64,
        dim_mults:      tuple = (1, 2, 4, 8),
        self_condition: bool  = True,
        with_time_emb:  bool  = True,
        controlnet:     bool  = False,
        concat_t2w:     bool  = False,
        img_channels:   int   = 1,
    ):
        super().__init__()

        # Basic configuration
        self.dim            = dim
        self.dim_mults      = tuple(dim_mults)
        self.self_condition = bool(self_condition)
        self.with_time_emb  = bool(with_time_emb)
        self.concat_t2w     = bool(concat_t2w)

        # Ensure modules can access channels
        self.input_img_channels = img_channels
        self.mask_channels      = img_channels
        
        # Channel composition: input image + condition + optional self-cond + optional t2w
        cond_channels           = img_channels
        self_cond_channels      = img_channels if self.self_condition else 0
        t2w_channels            = img_channels if self.concat_t2w else 0
        self.input_channels     = img_channels + cond_channels + self_cond_channels + t2w_channels  # channels fed into initial conv
        
        # Optional ControlNet 
        self.controlnet = ControlNet(dim, dim_mults, 1, self.input_channels, with_time_emb) if controlnet else None

        # Compute feature dimensions at each resolution and store pairs for building blocks
        dims        = [dim, *[dim * m for m in self.dim_mults]]
        self.in_out = list(zip(dims[:-1], dims[1:]))

        # Reusable conv block factory (ResNet block with 8 groups by default)
        self.block_factory = partial(ResnetBlock, groups=8)

        # Time embedding MLP
        if self.with_time_emb:
            self.time_dim = dim
            self.time_mlp = make_time_mlp(dim)
        else:
            self.time_dim = None
            self.time_mlp = None

         # Initial conv that maps concatenated inputs to base features
        self.init_conv = nn.Conv2d(self.input_channels, dim, kernel_size=7, padding=3)

        # Down and up stacks
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.num_resolutions = len(self.in_out)

        # ---- Downsampling path ---- #
        for i, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = i == (self.num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                self.block_factory(dim_in, dim_in, time_emb_dim=self.time_dim),
                self.block_factory(dim_in, dim_in, time_emb_dim=self.time_dim),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
            ]))

        # ---- Middle (bottleneck): ResNet -> cross-attn -> ResNet ---- #
        mid_dim = dims[-1]
        
        self.mid_block1     = self.block_factory(mid_dim, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn       = Residual(PreNorm(mid_dim, LinearCrossAttention(mid_dim)))
        self.mid_block2     = self.block_factory(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        # ---- Upsampling path (mirror of downs) ---- #
        for i, (dim_in, dim_out) in enumerate(reversed(self.in_out)):
            is_last = i == (self.num_resolutions - 1)
            
            self.ups.append(nn.ModuleList([
                self.block_factory(dim_in*3, dim_in, time_emb_dim = self.time_dim) if i < 3 else self.block_factory(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                self.block_factory(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))

        # ---- Final conv to map features back to image space ---- # (single-channel output by default) 
        self.final_res_block = self.block_factory(dim, dim, time_emb_dim = self.time_dim)
        self.final_conv      = nn.Conv2d(dim, img_channels, 1)

    def _apply_time(self, time):
        """Return time embedding or None if not used."""
        return self.time_mlp(time) if exists(self.time_mlp) else None

    def forward(self, x_img, low_res, time, x_self_cond=None, t2w=None, control=None):     
        """
        Forward pass:
        - x_img:        high-resolution input image tensor (B, C, H, W)
        - low_res:      conditioning low-res image (B, C, H, W)
        - time:         time-step tensor for diffusion (B,)
        - x_self_cond:  optional self-conditioning tensor (B, C, H, W)
        - t2w:          optional extra embedding 
        - control:      optional ControlNet conditioning
        """        
        assert x_img.shape == low_res.shape, 'Input different size to condition!'
        
        t_emb = self._apply_time(time)
            
        # Concatenate inputs: [x_img, low_res, (t2w), (self_condition)]
        parts = [x_img, low_res]
        if self.concat_t2w:
            assert t2w is not None, "concat_t2w=True but t2w is None"
            parts.append(t2w)
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x_img))
            parts.append(x_self_cond)
        x = torch.cat(parts, dim=1)
        
        # If using controlnet, add connections
        if self.controlnet is not None and control is not None:
            ctrl_res = iter(self.controlnet(x, control, t_emb))
            def add_ctrl(z): 
                return z + next(ctrl_res)
        else:
            def add_ctrl(z): 
                return z
            
        # Start UNet 
        x = self.init_conv(x)

        residuals = []   
        # ---- Downsampling path ---- #
        for conv1, conv2, downsample in self.downs:
            x = add_ctrl(conv1(x, t_emb));     residuals.append(x)
            x = add_ctrl(conv2(x, t_emb));     residuals.append(x)
            x = downsample(x)

        # --- Mid (bottleneck) --- #
        x = add_ctrl(self.mid_block1(x, t_emb))
        x = self.mid_attn(x)
        x = add_ctrl(self.mid_block2(x, t_emb))

        # --- Upsampling path --- #
        for conv1, conv2, upsample in self.ups:
            x = torch.cat((x, residuals.pop()), dim=1);     x = conv1(x, t_emb)
            x = torch.cat((x, residuals.pop()), dim=1);     x = conv2(x, t_emb)
            x = upsample(x)

        x = self.final_res_block(x, t_emb)
        return self.final_conv(x)

