import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from network_modules_DisC_Diff import timestep_embedding
from network_utils import default

class _CrossAttnResidual2D(nn.Module):
    """
    Bidirectional cross-attention in residual form:
      h_q <- h_q + gate * Attn(q=h_q, kv=h_kv)

    This module is *channel-dynamic*: it lazily creates an attention submodule
    for each encountered channel count C and caches it.
    """
    def __init__(self, num_heads=4, gate_init=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))  # scalar gate, tanh() applied

        # cache: key=str(C) -> nn.Module that implements attention for that C
        self._cache = nn.ModuleDict()

    def _build_for_channels(self, C: int) -> nn.Module:
        assert C % self.num_heads == 0, f"channels ({C}) must be divisible by num_heads ({self.num_heads})"

        # Use GroupNorm on channels directly; stable and cheap.
        # num_groups: clamp to a divisor of C, prefer 32.
        def _gn(c):
            g = 32
            while c % g != 0 and g > 1:
                g //= 2
            return nn.GroupNorm(num_groups=g, num_channels=c)

        block = nn.ModuleDict({
            "norm_q":  _gn(C),
            "norm_kv": _gn(C),

            # 1x1 convs are cheaper than Linear on flattened tokens and keep [B,C,H,W]
            "to_q": nn.Conv2d(C, C, kernel_size=1, bias=False),
            "to_k": nn.Conv2d(C, C, kernel_size=1, bias=False),
            "to_v": nn.Conv2d(C, C, kernel_size=1, bias=False),

            # output projection (equivalent to MHA out_proj)
            "to_out": nn.Conv2d(C, C, kernel_size=1, bias=False),
        })
        return block

    def forward(self, h_q: torch.Tensor, h_kv: torch.Tensor) -> torch.Tensor:
        """
        h_q, h_kv: [B, C, H, W]
        """
        assert h_q.shape == h_kv.shape, f"shape mismatch: {h_q.shape} vs {h_kv.shape}"
        B, C, H, W = h_q.shape

        key = str(C)
        if key not in self._cache:
            self._cache[key] = self._build_for_channels(C).to(device=h_q.device)

        blk = self._cache[key]

        # Project to q/k/v in [B,C,H,W]
        q  = blk["norm_q"](h_q)
        kv = blk["norm_kv"](h_kv)

        q_img  = blk["to_q"](q)
        k_img  = blk["to_k"](kv)
        v_img  = blk["to_v"](kv)

        # Reshape for SDPA: [B, heads, HW, head_dim]
        N = H * W
        hd = C // self.num_heads

        q_ = q_img.reshape(B, self.num_heads, hd, N).permute(0, 1, 3, 2)  # [B,h,N,hd]
        k_ = k_img.reshape(B, self.num_heads, hd, N).permute(0, 1, 3, 2)
        v_ = v_img.reshape(B, self.num_heads, hd, N).permute(0, 1, 3, 2)

        # Memory-efficient attention (no explicit weight tensor)
        def _attn_fn(q_, k_, v_):
            return F.scaled_dot_product_attention(q_, k_, v_, dropout_p=0.0, is_causal=False)
        
        if self.training:
            attn_out = torch_checkpoint(_attn_fn, q_, k_, v_)
        else:
            attn_out = _attn_fn(q_, k_, v_)

        # Back to [B,C,H,W]
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = blk["to_out"](attn_out)

        return h_q + torch.tanh(self.gate) * out



class UNet_MultiTask(nn.Module):
    """
    Wraps two UNet_Basic-like backbones and performs residual cross-attention exchange
    at each encoder block + middle + decoder block.

    Returns: (out_adc, out_t2w) with same spatial size.
    """
    def __init__(
        self,
        unet_adc: nn.Module,
        unet_t2w: nn.Module,
        *,
        num_heads:       int = 4,
        cross_encoder:   bool = True,
        cross_decoder:   bool = True,
        cross_middle:    bool = True,
        gate_init:       float = 0.0,
        cross_attention: bool = True
    ):
        super().__init__()
        self.adc = unet_adc
        self.t2w = unet_t2w
        self.cross_attention = cross_attention
        
        # Make it compatible with your Diffusion wrapper assumptions
        self.input_img_channels = 1
        self.mask_channels      = 1
        self.self_condition     = getattr(unet_adc, "self_condition", False) or getattr(unet_t2w, "self_condition", False)
        self.controlnet         = None
        self.use_T2W         = False

        # Cross-attn modules: one per block index (encoder + decoder), plus optional middle.
        self.cross_encoder = cross_encoder and cross_attention
        self.cross_decoder = cross_decoder and cross_attention
        self.cross_middle  = cross_middle  and cross_attention

        # Encoder blocks count must match for lockstep stepping
        assert len(self.adc.input_blocks)  == len(self.t2w.input_blocks),  "ADC and T2W UNets must have same number of input_blocks"
        assert len(self.adc.output_blocks) == len(self.t2w.output_blocks), "ADC and T2W UNets must have same number of output_blocks"

        if self.cross_attention:
            ## Attention modules are bidirectional, but we use separate modules so each direction can learn independently
            # Encoder
            self.xattn_enc_adc = nn.ModuleList([
                _CrossAttnResidual2D(num_heads=num_heads, gate_init=gate_init) for _ in range(len(self.adc.input_blocks))
            ])
            self.xattn_enc_t2w = nn.ModuleList([
                _CrossAttnResidual2D(num_heads=num_heads, gate_init=gate_init) for _ in range(len(self.t2w.input_blocks))
            ])
            
            # Middle
            self.xattn_mid_adc = _CrossAttnResidual2D(num_heads=num_heads, gate_init=gate_init)
            self.xattn_mid_t2w = _CrossAttnResidual2D(num_heads=num_heads, gate_init=gate_init)
            
            # Decoder 
            self.xattn_dec_adc = nn.ModuleList([
                _CrossAttnResidual2D(num_heads=num_heads, gate_init=gate_init) for _ in range(len(self.adc.output_blocks))
            ])
            self.xattn_dec_t2w = nn.ModuleList([
                _CrossAttnResidual2D(num_heads=num_heads, gate_init=gate_init) for _ in range(len(self.t2w.output_blocks))
            ])
        else:
            self.xattn_enc_adc = None
            self.xattn_enc_t2w = None
            self.xattn_mid_adc = None
            self.xattn_mid_t2w = None
            self.xattn_dec_adc = None
            self.xattn_dec_t2w = None
        

    def forward(
        self, x_adc, cond_adc, x_t2w, cond_t2w, timesteps, *, 
        x_self_cond_adc=None, x_self_cond_t2w=None, control=None
    ):
        """
        x_adc, x_t2w: noisy inputs for each task [B, 1, H, W]
        cond_adc, cond_t2w: conditioning tensors passed as 'low_res' argument [B, 1, H, W] (or matching channels)
        timesteps: [B]
        """
        # time embeddings per-branch (keeps parity if model_channels differ)
        emb_adc = self.adc.time_embed(timestep_embedding(timesteps, self.adc.model_channels))
        emb_t2w = self.t2w.time_embed(timestep_embedding(timesteps, self.t2w.model_channels))

        # Build initial concatenated inputs exactly like UNet_Basic does internally.
        if self.adc.self_condition:
            x_self_cond_adc = default(x_self_cond_adc, lambda: torch.zeros_like(x_adc))
            x_adc = torch.cat([x_adc, x_self_cond_adc], dim=1)
        
        if self.t2w.self_condition:
            x_self_cond_t2w = default(x_self_cond_t2w, lambda: torch.zeros_like(x_t2w))
            x_t2w = torch.cat([x_t2w, x_self_cond_t2w], dim=1)
              
        h_adc = torch.cat([x_adc, cond_adc], dim=1).type(self.adc.dtype)
        h_t2w = torch.cat([x_t2w, cond_t2w], dim=1).type(self.t2w.dtype)

        hs_adc = []
        hs_t2w = []
        
        # ----- Encoder lockstep -----
        for i, (blk_adc, blk_t2w) in enumerate(zip(self.adc.input_blocks, self.t2w.input_blocks)):
            h_adc = blk_adc(h_adc, emb_adc)
            h_t2w = blk_t2w(h_t2w, emb_t2w)

            if self.cross_encoder:
                h_adc = self.xattn_enc_adc[i](h_adc, h_t2w)
                h_t2w = self.xattn_enc_t2w[i](h_t2w, h_adc)

            hs_adc.append(h_adc)
            hs_t2w.append(h_t2w)

        # ----- Middle -----
        h_adc = self.adc.middle_block(h_adc, emb_adc)
        h_t2w = self.t2w.middle_block(h_t2w, emb_t2w)

        if self.cross_middle:
            h_adc = self.xattn_mid_adc(h_adc, h_t2w)
            h_t2w = self.xattn_mid_t2w(h_t2w, h_adc)

        # ----- Decoder lockstep -----
        for i, (blk_adc, blk_t2w) in enumerate(zip(self.adc.output_blocks, self.t2w.output_blocks)):
            h_adc = torch.cat([h_adc, hs_adc.pop()], dim=1)
            h_t2w = torch.cat([h_t2w, hs_t2w.pop()], dim=1)

            h_adc = blk_adc(h_adc, emb_adc)
            h_t2w = blk_t2w(h_t2w, emb_t2w)

            if self.cross_decoder:
                h_adc = self.xattn_dec_adc[i](h_adc, h_t2w)
                h_t2w = self.xattn_dec_t2w[i](h_t2w, h_adc)

        # Final heads
        h_adc = h_adc.type(x_adc.dtype)
        h_t2w = h_t2w.type(x_t2w.dtype)

        out_adc = self.adc.out(h_adc)
        out_t2w = self.t2w.out(h_t2w)
        return out_adc, out_t2w
