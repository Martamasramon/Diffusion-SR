import torch
from torch import nn
from functools import partial

from network_utils   import (
    exists, default,
    Downsample, Upsample
)
from network_modules import (
    ResnetBlock, Residual, PreNorm, LinearCrossAttention
)
from UNet_Basic import UNet_Basic


class UNet_Attn(UNet_Basic):
    """
    U-Net variant that inserts cross-attention layers in down and up paths.
    Inherits from UNet_Basic and replaces the down/ups modules with attention-enabled blocks.
    """
    def __init__(self, *args, use_T2W=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_T2W = bool(use_T2W)

        # Recreate the down/ups stacks to include cross-attention blocks
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        
        # Figure out size of context vectors for cross attention
        t2w_channels_by_level = [1, 64, 128, 256]
        
        # Compute context channels for each down level (0 if T2W not used)
        context_channels = [(t2w_channels_by_level[i] if self.use_T2W else 0) for i in range(len(self.in_out))]

        # Build down path with an additional cross-attention block per resolution
        for i, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = i == (self.num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                self.block_factory(dim_in, dim_in, time_emb_dim = self.time_dim),
                self.block_factory(dim_in, dim_in, time_emb_dim = self.time_dim),
                # Add cross-attention block that can accept context vectors sized by context_channels[i]
                Residual(PreNorm(dim_in, LinearCrossAttention(dim_in, context_in=context_channels[i]))), ## <----
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        # Build up path: attention blocks mirrored and using reversed context channels
        for i, (dim_in, dim_out) in enumerate(reversed(self.in_out)):
            is_last = i == (self.num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                self.block_factory(dim_in*3, dim_in, time_emb_dim = self.time_dim) if i < 3 else self.block_factory(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                self.block_factory(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                # Add cross-attention block that can accept context vectors sized by context_channels[i]
                Residual(PreNorm(dim_in, LinearCrossAttention(dim_in, context_in=context_channels[::-1][i]))), ## <---- 
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))


    def forward(self, x_img, low_res, time, x_self_cond=None, t2w=None, control=None):   
        """
        Forward follows base U-Net logic with extra cross-attention ops included.
        - t2w must be provided if use_T2W is True.
        """
        assert not (self.use_T2W and t2w is None), "T2W embedding required but not provided"
            
        t_emb = self._apply_time(time)
            
        # Concatenate inputs: [x_img, low_res, (self_condition)]
        parts = [x_img, low_res]
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x_img))
            parts.append(x_self_cond)
        x = torch.cat(parts, dim=1)
    
        x = self.init_conv(x)
        
        # ---- Downsampling path with cross attention ---- #
        residuals = []
        for i, (conv1, conv2, cross_attention, downsample) in enumerate(self.downs):
            context = t2w[i]
            x = conv1(x, t_emb); residuals.append(x)
            x = conv2(x, t_emb); 
            x = cross_attention(x, context); residuals.append(x)
            x = downsample(x)

        # --- Mid (bottleneck) --- #
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # --- Upsampling path with cross attention --- #
        for i, (conv1, conv2, cross_attention, upsample) in enumerate(self.ups):
            context = t2w[-(i+1)]
            x = torch.cat((x, residuals.pop()), dim=1); x = conv1(x, t_emb)
            x = torch.cat((x, residuals.pop()), dim=1); x = conv2(x, t_emb)
            x = cross_attention(x, context)
            x = upsample(x)

        x = self.final_res_block(x, t_emb)
        return self.final_conv(x) 

