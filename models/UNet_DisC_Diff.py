import torch 
import torch.nn as nn
import copy

from network_utils   import default
from network_modules_DisC_Diff import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
    TimestepEmbedSequential,
    SE_Attention,
    Upsample,
    Downsample,
    ResBlock,
    AttentionBlock,
    convert_module_to_f16, 
    convert_module_to_f32
)
        
class UNet_Basic(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param model_channels: base channel count for the model.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: downsample rates at which attention will take place. 
        May be a set, list, or tuple. (i.e. if contains 4, at 4x downsampling attention will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use fixed channel width.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
            self,
            image_size              = 64,
            use_T2W                 = False,
            attention_resolutions   = (2, 4, 8),     # (4, 8, 16) if 128
            channel_mult            = (1, 2, 4, 8), # (1, 2, 4, 8, 8) if 128
            model_channels          = 96,
            num_res_blocks          = 2,
            dropout                 = 0,
            conv_resample           = True,
            self_condition          = True, # False if latent!
            dims                    = 2,
            use_checkpoint          = False,
            use_fp16                = False,
            num_heads               = 4,
            num_head_channels       = 48,
            use_scale_shift_norm    = False,
            resblock_updown         = False,
            in_channels             = None
    ):
        super().__init__()

        self.image_size         = image_size
        self.in_channels        = (3 if use_T2W else 2) if in_channels is None else in_channels
        self.in_channels        = self.in_channels + 1 if self_condition else self.in_channels
        self.model_channels     = model_channels
        self.out_channels       = 1 
        self.num_res_blocks     = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout            = dropout
        self.channel_mult       = channel_mult
        self.conv_resample      = conv_resample
        self.use_checkpoint     = use_checkpoint
        self.dtype              = torch.float16 if use_fp16 else torch.float32
        self.num_heads          = num_heads
        self.num_head_channels  = num_head_channels
        self.num_heads_upsample = num_heads
        self.dims               = dims
        self.self_condition     = self_condition
        
        # Make compatible with diffusion script
        self.input_img_channels = 1
        self.mask_channels      = 1
        self.controlnet         = None
        self.concat_t2w         = use_T2W

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = self.ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(self.dims , self.in_channels, ch, 3, padding=1))]
        )

        # Input (downstream) blocks
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=self.dims ,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=self.dims ,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=self.dims , out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
                
        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=self.dims ,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=self.dims ,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Output (upstream) blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=self.dims ,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=self.dims ,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=self.dims , out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims , ch, self.out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, low_res, timesteps, x_self_cond=None, t2w=None, control=None):
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat([x, x_self_cond], dim=1)
        
        # concatenate inputs 
        if t2w is not None:
            h = torch.cat([x, low_res, t2w], dim=1).type(self.dtype)
        else:
            h = torch.cat([x, low_res], dim=1).type(self.dtype)

        # encoder + skips
        for idx in range(len(self.input_blocks)):
            h = self.input_blocks[idx](h, emb)
            hs.append(h)

        # bottleneck
        h = self.middle_block(h, emb)

        # decoder
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)


     
##########################################################################################################################################################################
##########################################################################################################################################################################


class UNet_DisC_Diff(UNet_Basic):
    """
    The full UNet model with attention and timestep embedding.
    :param model_channels: base channel count for the model.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: downsample rates at which attention will take place. 
        May be a set, list, or tuple. (i.e. if contains 4, at 4x downsampling attention will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use fixed channel width.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(self, *args, in_channels=1, **kwargs):
        super().__init__(*args, in_channels=in_channels, **kwargs)    
        
        conv_ch = self.image_size # 288

        self.input_blocks_lr  = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_t2w = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])

        enc_ch = int(self.model_channels * self.channel_mult[-1])

        self.SE_Attention_com    = SE_Attention(channel=enc_ch//2, reduction=8)
        self.SE_Attention_dist_1 = SE_Attention(channel=enc_ch//2, reduction=8)
        self.SE_Attention_dist_2 = SE_Attention(channel=enc_ch//2, reduction=8)
        self.SE_Attention_dist_3 = SE_Attention(channel=enc_ch//2, reduction=8)

        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(self.dims , 2 * enc_ch, enc_ch, 1, padding=0),
            nn.SiLU()
        )

        self.conv_common = nn.Sequential(
            conv_nd(self.dims , enc_ch, enc_ch//2, 3, padding=1),
            nn.SiLU()
        )

        self.conv_distinct = nn.Sequential(
            conv_nd(self.dims , enc_ch, enc_ch//2, 3, padding=1),
            nn.SiLU()
        )

    def forward(self, x, low_res, timesteps,  x_self_cond=None, t2w=None, control=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h1 = x.type(self.dtype)
        h2 = low_res.type(self.dtype)
        h3 = t2w.type(self.dtype)

        for idx in range(len(self.input_blocks)):
            h1 = self.input_blocks[idx](h1, emb)
            h2 = self.input_blocks_lr[idx](h2, emb)
            h3 = self.input_blocks_t2w[idx](h3, emb)
            hs.append((1 / 3) * h1 + (1 / 3) * h2 + (1 / 3) * h3)

        com_h1 = self.conv_common(h1)
        com_h2 = self.conv_common(h2)

        dist_h1 = self.conv_distinct(h1)
        dist_h2 = self.conv_distinct(h2)

        dist_h1 = self.SE_Attention_dist_1(dist_h1)
        dist_h2 = self.SE_Attention_dist_2(dist_h2)


        com_h3 = self.conv_common(h3)
        dist_h3 = self.conv_distinct(h3)
        com_h = self.SE_Attention_com((1 / 3) * com_h1 + (1 / 3) * com_h2 + (1 / 3) * com_h3)
        dist_h3 = self.SE_Attention_dist_3(dist_h3)
        h = torch.cat([com_h, dist_h1, dist_h2, dist_h3], dim=1)
        h = self.dim_reduction_non_zeros(h)

        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        return self.out(h)
