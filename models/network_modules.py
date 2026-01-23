# Realtive import
import sys
sys.path.append('../blocks')
import torch
import torch.nn.functional as F
from torch import nn, einsum
import math
import numpy as np
from functools     import partial
from network_utils import *
from einops        import rearrange,repeat


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
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

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
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
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
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

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
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
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
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, context=None):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class LinearCrossAttention(nn.Module):
    def __init__(self, dim, context_in=None, heads = 4, dim_head = 32):
        super(LinearCrossAttention, self).__init__()

        context_in = default(context_in, dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Conv2d(context_in, hidden_dim, 1, bias = False)
        self.to_v = nn.Conv2d(context_in, hidden_dim, 1, bias = False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # b, c, h, w = x.shape
        # qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), (q, k, v))
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        
        return self.to_out(out)

class CoreEnhance(nn.Module):
    def __init__(self, iC_list, iSize_list, out_c):
        super(CoreEnhance, self).__init__()
        ic0, ic1, ic2, ic3 = iC_list
        self.iS0, self.iS1, self.iS2, self.iS3 = iSize_list
        # 1 column
        self.col13 = ConvBlock(out_c, out_c)
        self.col12 = ConvBlock(out_c + out_c, out_c)
        self.col11 = ConvBlock(out_c + out_c, out_c)
        self.col10 = ConvBlock(out_c + out_c + out_c, out_c)

        # 2 column
        self.col23 = ConvBlock(out_c, out_c)
        self.col21 = ConvBlock(out_c + out_c, out_c)
        self.col20 = ConvBlock(out_c + out_c + out_c, out_c)

        # 3 column
        self.col33 = ConvBlock(out_c, out_c)
        self.col32 = ConvBlock(out_c + out_c + out_c, out_c)

        # 4 column
        self.col43 = ConvBlock(out_c, out_c)

    def forward(self, xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        side_out = []
        stage4 = xs[-1]
        stage40 = self.col13(stage4)
        stage41 = self.col23(stage40)
        stage42 = self.col33(stage41)
        stage43 = self.col43(stage42)
        side_out.append(stage43)
        stage40 = F.interpolate(stage40, size=(self.iS1,self.iS1), mode='bilinear')
        stage41 = F.interpolate(stage41, size=(self.iS1,self.iS1), mode='bilinear')
        stage42 = F.interpolate(stage42, size=(self.iS1,self.iS1), mode='bilinear')
        stage43 = F.interpolate(stage43, size=(self.iS1,self.iS1), mode='bilinear')

        stage31 = self.col12(torch.cat((stage40, xs[-2]), dim=1))
        stage32 = self.col21(torch.cat((stage41, stage31), dim=1))
        stage33 = self.col32(torch.cat((stage42, stage43, stage32), dim=1))
        side_out.append(stage33)
        stage31 = F.interpolate(stage31, size=(self.iS2,self.iS2), mode='bilinear')
        stage32 = F.interpolate(stage32, size=(self.iS2,self.iS2), mode='bilinear')
        stage33 = F.interpolate(stage33, size=(self.iS2,self.iS2), mode='bilinear')

        stage21 = self.col11(torch.cat((stage31, xs[-3]), dim=1))
        stage22 = self.col20(torch.cat((stage32, stage33, stage21), dim=1))
        side_out.append(stage22)
        stage21 = F.interpolate(stage21, size=(self.iS3,self.iS3), mode='bilinear')
        stage22 = F.interpolate(stage22, size=(self.iS3,self.iS3), mode='bilinear')

        stage1 = self.col10(torch.cat((stage21, stage22, xs[-4]), dim=1))
        side_out.append(stage1)

        return stage1, side_out

class BoundaryEnhance(nn.Module):
    def __init__(self, iC_list, iSize_list, out_c):
        super(BoundaryEnhance, self).__init__()
        ic0, ic1, ic2, ic3 = iC_list
        self.iS0, self.iS1, self.iS2, self.iS3 = iSize_list
        # 1 column
        self.col13 = ConvBlock(out_c, out_c)
        self.col12 = ConvBlock(out_c + out_c, out_c)
        self.col11 = ConvBlock(out_c + out_c, out_c)
        self.col10 = ConvBlock(out_c + out_c, out_c)

        # 2 column
        self.col22 = ConvBlock(out_c + out_c, out_c)
        self.col21 = ConvBlock(out_c + out_c, out_c)
        self.col20 = ConvBlock(out_c + out_c, out_c)

        # 3 column
        self.col31 = ConvBlock(out_c + out_c, out_c)
        self.col30 = ConvBlock(out_c + out_c, out_c)

        # 4 column
        self.col40 = ConvBlock(out_c + out_c, out_c)

    def forward(self, xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        side_out = []
        stage4 = xs[-1]
        stage4 = self.col13(stage4)
        side_out.append(stage4)
        stage4 = F.interpolate(stage4, size=(self.iS1,self.iS1), mode='bilinear')
        stage31 = self.col12(torch.cat((stage4, xs[-2]), dim=1))
        stage32 = self.col22(torch.cat((stage4, stage31), dim=1))
        side_out.append(stage32)

        stage31 = F.interpolate(stage31, size=(self.iS2,self.iS2), mode='bilinear')
        stage32 = F.interpolate(stage32, size=(self.iS2,self.iS2), mode='bilinear')
        stage21 = self.col11(torch.cat((stage31, xs[-3]), dim=1))
        stage22 = self.col21(torch.cat((stage32, stage21), dim=1))
        stage23 = self.col31(torch.cat((stage32, stage22), dim=1))
        side_out.append(stage23)

        stage21 = F.interpolate(stage21, size=(self.iS3,self.iS3), mode='bilinear')
        stage22 = F.interpolate(stage22, size=(self.iS3,self.iS3), mode='bilinear')
        stage23 = F.interpolate(stage23, size=(self.iS3,self.iS3), mode='bilinear')
        stage11 = self.col10(torch.cat((stage21, xs[-4]), dim=1))
        stage12 = self.col20(torch.cat((stage22, stage11), dim=1))
        stage13 = self.col30(torch.cat((stage23, stage12), dim=1))
        stage14 = self.col40(torch.cat((stage23, stage13), dim=1))
        side_out.append(stage14)

        return stage14, side_out

class DecoderCoreBoundary(nn.Module):
    def __init__(self,out_c):
        super(DecoderCoreBoundary, self).__init__()
        self.conv0 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(out_c)
        self.ConvBlock0 = ConvBlock(out_c, out_c)

        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.ConvBlock1 = ConvBlock(out_c, out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.ConvBlock2 = ConvBlock(out_c, out_c)

        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        # self.ConvBlock3 = ConvBlock(64, 32)


    def forward(self, input1, input2):
        out_list = []
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out_list.append(out0)

        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out0 = self.ConvBlock0(out0)
        out1 = F.relu(self.bn1(self.conv1(input1[1] + input2[1] + out0)), inplace=True)
        out_list.append(out1)

        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out1 = self.ConvBlock1(out1)
        out2 = F.relu(self.bn2(self.conv2(input1[2] + input2[2] + out1)), inplace=True)
        out_list.append(out2)

        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out2 = self.ConvBlock2(out2)
        out3 = F.relu(self.bn3(self.conv3(input1[3] + input2[3] + out2)), inplace=True)
        out_list.append(out3)
        return out_list

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context=None):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class ConvBlock(nn.Module):
    def  __init__(self, dim, dim_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(dim_out)
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
            nn.Conv2d(C_in, C_in, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
    
    

class BodyEnhance(nn.Module):
    def __init__(self, iC_list, iSize_list, out_c):
        super(BodyEnhance, self).__init__()
        ic0, ic1, ic2, ic3 = iC_list
        self.iS0, self.iS1, self.iS2, self.iS3 = iSize_list
        # 1 column
        self.col13 = ConvBlock(out_c, out_c)
        self.col12 = ConvBlock(out_c + out_c, out_c)
        self.col11 = ConvBlock(out_c + out_c, out_c)
        self.col10 = ConvBlock(out_c + out_c + out_c, out_c)

        # 2 column
        self.col23 = ConvBlock(out_c, out_c)
        self.col21 = ConvBlock(out_c + out_c, out_c)
        self.col20 = ConvBlock(out_c + out_c + out_c, out_c)

        # 3 column
        self.col33 = ConvBlock(out_c, out_c)
        self.col32 = ConvBlock(out_c + out_c + out_c, out_c)

        # 4 column
        self.col43 = ConvBlock(out_c, out_c)

    def forward(self, xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        side_out = []
        stage4 = xs[-1]
        stage40 = self.col13(stage4)
        stage41 = self.col23(stage40)
        stage42 = self.col33(stage41)
        stage43 = self.col43(stage42)
        side_out.append(stage43)
        stage40 = F.interpolate(stage40, size=(self.iS1,self.iS1), mode='bilinear')
        stage41 = F.interpolate(stage41, size=(self.iS1,self.iS1), mode='bilinear')
        stage42 = F.interpolate(stage42, size=(self.iS1,self.iS1), mode='bilinear')
        stage43 = F.interpolate(stage43, size=(self.iS1,self.iS1), mode='bilinear')

        stage31 = self.col12(torch.cat((stage40, xs[-2]), dim=1))
        stage32 = self.col21(torch.cat((stage41, stage31), dim=1))
        stage33 = self.col32(torch.cat((stage42, stage43, stage32), dim=1))
        side_out.append(stage33)
        stage31 = F.interpolate(stage31, size=(self.iS2,self.iS2), mode='bilinear')
        stage32 = F.interpolate(stage32, size=(self.iS2,self.iS2), mode='bilinear')
        stage33 = F.interpolate(stage33, size=(self.iS2,self.iS2), mode='bilinear')

        stage21 = self.col11(torch.cat((stage31, xs[-3]), dim=1))
        stage22 = self.col20(torch.cat((stage32, stage33, stage21), dim=1))
        side_out.append(stage22)
        stage21 = F.interpolate(stage21, size=(self.iS3,self.iS3), mode='bilinear')
        stage22 = F.interpolate(stage22, size=(self.iS3,self.iS3), mode='bilinear')

        stage1 = self.col10(torch.cat((stage21, stage22, xs[-4]), dim=1))
        side_out.append(stage1)

        return stage1, side_out

class DetailEnhance(nn.Module):
    def __init__(self, iC_list, iSize_list, out_c):
        super(DetailEnhance, self).__init__()
        ic0, ic1, ic2, ic3 = iC_list
        self.iS0, self.iS1, self.iS2, self.iS3 = iSize_list
        # 1 column
        self.col13 = ConvBlock(out_c, out_c)
        self.col12 = ConvBlock(out_c + out_c, out_c)
        self.col11 = ConvBlock(out_c + out_c, out_c)
        self.col10 = ConvBlock(out_c + out_c, out_c)

        # 2 column
        self.col22 = ConvBlock(out_c + out_c, out_c)
        self.col21 = ConvBlock(out_c + out_c, out_c)
        self.col20 = ConvBlock(out_c + out_c, out_c)

        # 3 column
        self.col31 = ConvBlock(out_c + out_c, out_c)
        self.col30 = ConvBlock(out_c + out_c, out_c)

        # 4 column
        self.col40 = ConvBlock(out_c + out_c, out_c)

    def forward(self, xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        side_out = []
        stage4 = xs[-1]
        stage4 = self.col13(stage4)
        side_out.append(stage4)
        stage4 = F.interpolate(stage4, size=(self.iS1,self.iS1), mode='bilinear')
        stage31 = self.col12(torch.cat((stage4, xs[-2]), dim=1))
        stage32 = self.col22(torch.cat((stage4, stage31), dim=1))
        side_out.append(stage32)

        stage31 = F.interpolate(stage31, size=(self.iS2,self.iS2), mode='bilinear')
        stage32 = F.interpolate(stage32, size=(self.iS2,self.iS2), mode='bilinear')
        stage21 = self.col11(torch.cat((stage31, xs[-3]), dim=1))
        stage22 = self.col21(torch.cat((stage32, stage21), dim=1))
        stage23 = self.col31(torch.cat((stage32, stage22), dim=1))
        side_out.append(stage23)

        stage21 = F.interpolate(stage21, size=(self.iS3,self.iS3), mode='bilinear')
        stage22 = F.interpolate(stage22, size=(self.iS3,self.iS3), mode='bilinear')
        stage23 = F.interpolate(stage23, size=(self.iS3,self.iS3), mode='bilinear')
        stage11 = self.col10(torch.cat((stage21, xs[-4]), dim=1))
        stage12 = self.col20(torch.cat((stage22, stage11), dim=1))
        stage13 = self.col30(torch.cat((stage23, stage12), dim=1))
        stage14 = self.col40(torch.cat((stage23, stage13), dim=1))
        side_out.append(stage14)

        return stage14, side_out

class Decoder_body_detail(nn.Module):
    def __init__(self,in_c, out_c):
        super(Decoder_body_detail, self).__init__()
        self.conv0 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(out_c)
        self.ConvBlock0 = ConvBlock(out_c, out_c)

        self.conv1 = nn.Conv2d(in_c + out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.ConvBlock1 = ConvBlock(out_c, out_c)

        self.conv2 = nn.Conv2d(in_c + out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.ConvBlock2 = ConvBlock(out_c, out_c)

        self.conv3 = nn.Conv2d(in_c + out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        # self.ConvBlock3 = ConvBlock(64, 32)


    def forward(self, input1, input2):
        out_list = []
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out_list.append(out0)

        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out0 = self.ConvBlock0(out0)
        out1 = torch.cat(((input1[1] + input2[1]), out0), dim=1)
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out_list.append(out1)

        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out1 = self.ConvBlock1(out1)
        out2 = torch.cat(((input1[2] + input2[2]), out1), dim=1)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out_list.append(out2)

        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out2 = self.ConvBlock2(out2)
        out3 = torch.cat(((input1[3] + input2[3]), out2), dim=1)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out_list.append(out3)
        return out_list


class Decoder_detail(nn.Module):
    def __init__(self):
        super(Decoder_detail, self).__init__()
        self.conv0 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # self.FEM1 = Frequency_Edge_Module(radius=16, channel=64)
        # self.FEM2 = Frequency_Edge_Module(radius=16, channel=128)


    def forward(self, input1, input2=[0, 0, 0, 0]):
        out_list = []
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out_list.append(out0)

        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(  torch.cat((input1[1],out0), dim=1) )), inplace=True)
        out_list.append(out1)


        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        # input12 = self.FEM2(input1[2])
        out2 = F.relu(self.bn2(self.conv2(   torch.cat((input1[2],out1), dim=1 ))), inplace=True)
        out_list.append(out2)


        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        # input13 = self.FEM1(input1[3])
        out3 = F.relu(self.bn3(self.conv3(  torch.cat((input1[3],out2), dim=1 ))), inplace=True)
        out_list.append(out3)
        return out_list

class Decoder_body(nn.Module):
    def __init__(self):
        super(Decoder_body, self).__init__()
        self.conv0 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0, 0, 0, 0]):
        out_list = []
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out_list.append(out0)

        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(torch.cat((input1[1], out0), dim=1))), inplace=True)
        out_list.append(out1)

        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(torch.cat((input1[2], out1), dim=1))), inplace=True)
        out_list.append(out2)

        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(torch.cat((input1[3], out2), dim=1))), inplace=True)
        out_list.append(out3)
        return out_list
 

class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, dilation, padding):
        super(DWConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding, groups=in_channel,
                                dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.DWConv(x)
        out = self.selu(self.bn(x))

        return out


class DWSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding, kernels_per_layer):
        super(DWSConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, in_channel * kernels_per_layer, kernel_size=kernel, padding=padding,
                                groups=in_channel, bias=False)
        self.bn = nn.BatchNorm2d(in_channel * kernels_per_layer)
        self.selu = nn.SELU()
        self.PWConv = nn.Conv2d(in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

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
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))

        self.norm_input = LayerNorm(dim, bias=True)
        self.norm_condition = LayerNorm(dim, bias=True)

        self.block = ResnetBlock(dim, dim)

    def forward(self, x, c):
        # ff-parser in the paper, for modulating out the high frequencies

        dtype = x.dtype
        x = fft2(x)
        x = x * self.ff_parser_attn_map
        x = ifft2(x).real
        x = x.type(dtype)

        # eq 3 in paper

        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # add an extra block to allow for more integration of information
        # there is a downsample right after the Condition block (but maybe theres a better place to condition than right before the downsample)

        return self.block(c)
    
    
class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1, padding=0)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
        

class Identity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x
    
        
def make_time_mlp(dim: int) -> nn.Module:
    return nn.Sequential(
        SinusoidalPosEmb(dim),
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Linear(dim * 4, dim),
    )