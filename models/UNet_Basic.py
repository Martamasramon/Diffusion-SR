import torch
from torch      import nn
from functools  import partial
import numpy    as np

import os
from network_utils   import *
from network_modules import *  
from ControlNet      import ControlNet

class UNet_Basic(nn.Module):
    def __init__(
        self,
        dim             = 64,
        dim_mults       = (1, 2, 4, 8),
        self_condition  = True,
        with_time_emb   = True,
        controlnet      = False,
        concat_t2w      = False,
        img_channels    = 1
    ):
        super().__init__()
        
        # determine dimensions
        self.dim                = dim
        self.dim_mults          = dim_mults
        self.self_condition     = self_condition
        self.with_time_emb      = with_time_emb

        self.input_img_channels = img_channels
        self.mask_channels      = self.input_img_channels
        cond_channels           = self.input_img_channels
        self_cond_channels      = self.input_img_channels if self_condition else 0
        t2w_channels            = self.input_img_channels if concat_t2w else 0 
        input_channels          = self.mask_channels + cond_channels + self_cond_channels + t2w_channels
        self.controlnet         = ControlNet(dim, dim_mults, 1, input_channels, with_time_emb) if controlnet else None
        self.concat_t2w         = concat_t2w

        dims             = [dim, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        self.block_klass = partial(ResnetBlock, groups = 8)

        # time embedding 
        if with_time_emb:
            self.time_dim = dim
            self.time_mlp = make_time_mlp(dim)
        else:
            self.time_dim = None
            self.time_mlp = None

        self.init_conv          = nn.Conv2d(input_channels, dim, 7, padding = 3)
        self.downs_label_noise  = nn.ModuleList([])
        self.ups                = nn.ModuleList([])

        self.num_resolutions = len(self.in_out)

        # ---- DOWN ---- #
        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (self.num_resolutions - 1)

            self.downs_label_noise.append(nn.ModuleList([
                self.block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                self.block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),

                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        # ---- MID ---- #
        mid_dim = dims[-1]
        
        self.mid_block1     = self.block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn       = Residual(PreNorm(mid_dim, LinearCrossAttention(mid_dim)))
        self.mid_block2     = self.block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        # ---- UP ---- #
        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out)):
            is_last = ind >= (self.num_resolutions - 1)
            
            self.ups.append(nn.ModuleList([
                self.block_klass(dim_in*3, dim_in, time_emb_dim = self.time_dim) if ind < 3 else self.block_klass(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                self.block_klass(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))

        # ---- FINAL ---- #
        self.final_res_block = self.block_klass(dim, dim, time_emb_dim = self.time_dim)
        self.final_conv      = nn.Sequential(
            nn.Conv2d(dim, self.mask_channels, 1),
        )


    def normalization(channels):
        """
        Make a standard normalization layer.

        :param channels: number of input channels.
        :return: an nn.Module for normalization.
        """
        return GroupNorm32(32, channels)


    def forward(self, input_x, low_res, time, x_self_cond=None, t2w=None, control=None):     
        assert input_x.shape == low_res.shape, 'Input different size to condition!'

        B,C,H,W, = input_x.shape
        if low_res is not None:
            x = torch.cat((input_x, low_res, t2w), dim=1) if t2w is not None else torch.cat((input_x, low_res), dim=1)
        
        if self.self_condition: # could use 0.5 probability of using self-condition?
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(input_x))
            x = torch.cat((x, x_self_cond), dim=1)
        
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        
        if (control is not None) and (self.controlnet is not None):
            ctrl_res = iter(self.controlnet(x, control, t))
            def add_ctrl(z): 
                return z + next(ctrl_res)
        else:
            def add_ctrl(z): 
                return z

        x = self.init_conv(x)
        residuals = []   
             
        # ---- DOWN ---- #
        for conv1, conv2, downsample in self.downs_label_noise:
            x = add_ctrl(conv1(x, t));     residuals.append(x)
            x = add_ctrl(conv2(x, t));     residuals.append(x)
            x = downsample(x)

        # --- MID --- #
        x = add_ctrl(self.mid_block1(x, t))
        x = self.mid_attn(x)
        x = add_ctrl(self.mid_block2(x, t))

        # --- UP --- #
        for conv1, conv2, upsample in self.ups:
            x = torch.cat((x, residuals.pop()), dim=1);     x = conv1(x, t)
            x = torch.cat((x, residuals.pop()), dim=1);     x = conv2(x, t)
            x = upsample(x)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

