# Realtive import
import sys
sys.path.append('../blocks')
import torch
import torch.nn.functional as F
from torch import nn, einsum
import math
from network_utils import default, exists, divisible_by
from einops import rearrange

spatial_dims = 3  # or 2
ConvND        = nn.Conv3d            if spatial_dims == 3 else nn.Conv2d
BatchNormND   = nn.BatchNorm3d       if spatial_dims == 3 else nn.BatchNorm2d
AdaptiveAvgND = nn.AdaptiveAvgPool3d if spatial_dims == 3 else nn.AdaptiveAvgPool2d
AdaptiveMaxND = nn.AdaptiveMaxPool3d if spatial_dims == 3 else nn.AdaptiveMaxPool2d
mode = "trilinear" if spatial_dims == 3 else "bilinear"

class EMA():
    def __init__(self, beta):
        super(EMA, self).__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
    
class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, context=None, *args, **kwargs):
        return self.fn(x, context, *args, **kwargs) + x


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones((1, dim) + (1,)*spatial_dims))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = ConvND(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = ConvND(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.size(0), time_emb.size(1), *([1]*spatial_dims))
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones((1, dim) + (1,)*spatial_dims))
        self.b = nn.Parameter(torch.zeros((1, dim) + (1,)*spatial_dims))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.norm(x)
        return self.fn(x,context)

class GlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view((x.size(0), x.size(1)) + (1,)*spatial_dims)

class BasicConvND(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConvND, self).__init__()
        self.conv = ConvND(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = BatchNormND(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = AdaptiveAvgPoolND(1)
        self.max_pool = AdaptiveMaxPoolND(1)

        self.fc = nn.Sequential(ConvND(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                ConvND(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = ConvND(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = ConvND(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            ConvND(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, context=None):
        b, c, *spatial = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) ... -> b h c n', h=self.heads), (q, k, v))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.view(b, self.heads * (out.shape[2]), *spatial)
        return self.to_out(out)

class LinearCrossAttention(nn.Module):
    def __init__(self, dim, context_in=None, heads = 4, dim_head = 32):
        super(LinearCrossAttention, self).__init__()

        context_in = default(context_in, dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = ConvND(dim, hidden_dim, 1, bias = False)
        self.to_k = ConvND(context_in, hidden_dim, 1, bias = False)
        self.to_v = ConvND(context_in, hidden_dim, 1, bias = False)

        self.to_out = ConvND(hidden_dim, dim, 1)

    def forward(self, x, context=None, mask=None):
        b, c, *spatial = x.shape
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # b, c, h, w = x.shape
        # qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) ... -> b h c n', h=self.heads), (q, k, v))
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.view(b, self.heads * (out.shape[2]), *spatial)
        
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = ConvND(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = ConvND(hidden_dim, dim, 1)

    def forward(self, x, context=None):
        b, c, *spatial = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) ... -> b h c n', h=self.heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = out.view(b, self.heads * (out.shape[-1]), *spatial)
        return self.to_out(out)

class ConvBlock(nn.Module):
    def  __init__(self, dim, dim_out):
        super(ConvBlock, self).__init__()
        self.conv = ConvND(dim, dim_out, kernel_size=3, stride=1, padding=1)
        self.bn = BatchNormND(dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.relu(h)
        return h

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            ConvND(C_in, C_in, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=False),
            ConvND(C_in, C_out, kernel_size=1, padding=0, bias=False),
            BatchNormND(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
    

class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, dilation, padding):
        super(DWConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = ConvND(in_channel, out_channel, kernel_size=kernel, padding=padding, groups=in_channel,
                                dilation=dilation, bias=False)
        self.bn = BatchNormND(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.DWConv(x)
        out = self.selu(self.bn(x))

        return out


class DWSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding, kernels_per_layer):
        super(DWSConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = ConvND(in_channel, in_channel * kernels_per_layer, kernel_size=kernel, padding=padding,
                                groups=in_channel, bias=False)
        self.bn = BatchNormND(in_channel * kernels_per_layer)
        self.selu = nn.SELU()
        self.PWConv = ConvND(in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False)
        self.bn2 = BatchNormND(out_channel)

    def forward(self, x):
        x = self.DWConv(x)
        x = self.selu(self.bn(x))
        out = self.PWConv(x)
        out = self.selu(self.bn2(out))

        return out
    

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            heads=4,
            depth=1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, dim_head=dim_head, heads=heads)),
                Residual(FeedForward(dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class Conditioning(nn.Module):
    def __init__(self, fmap_size, dim):
        super().__init__()
        # for 3D: fmap_size must be a tuple (D,H,W); for 2D it can remain int or (H,W)
        map_shape = (dim,) + (fmap_size if isinstance(fmap_size, tuple) else (fmap_size,)*spatial_dims)
        self.ff_parser_attn_map = nn.Parameter(torch.ones(map_shape))

        self.norm_input = LayerNorm(dim, bias=True)
        self.norm_condition = LayerNorm(dim, bias=True)

        self.block = ResnetBlock(dim, dim)

    def forward(self, x, c):
        # ff-parser in the paper, for modulating out the high frequencies

        dtype = x.dtype
        x = torch.fft.fftn(x, dim=tuple(range(-spatial_dims, 0)))
        x = x * self.ff_parser_attn_map
        x = torch.fft.ifftn(x, dim=tuple(range(-spatial_dims, 0))).real
        x = x.type(dtype)

        # eq 3 in paper

        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # add an extra block to allow for more integration of information
        # there is a downsample right after the Condition block (but maybe theres a better place to condition than right before the downsample)

        return self.block(c)
    
    
class ZeroConvND(ConvND):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1, padding=0)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
        