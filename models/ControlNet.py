from torch      import nn
from functools  import partial
from network_modules import ResnetBlock, Residual, PreNorm, LinearCrossAttention, ZeroConv2d
from network_utils   import Downsample

class ControlNet(nn.Module):
    """
    Produces a list of residual tensors that align with UNet_Basic's injection sites:
      - 2 residuals per encoder level (after each ResBlock)
      - 2 residuals at the mid (after block1 and block2)
      - 2 residuals per decoder level (after each ResBlock)
      - 1 residual at the final ResBlock (optional but handy)
    """
    def __init__(
        self, 
        dim                 = 64, 
        dim_mults           = (1,2,4,8), 
        control_in_channels = 1, 
        main_input_channels = 1,
        with_time_emb       = True
    ):
        super().__init__()
        
        self.dim            = dim
        self.dim_mults      = dim_mults
        self.block_klass    = partial(ResnetBlock, groups=8)

        self.init_control_conv = ZeroConv2d(control_in_channels, dim)  
        self.init_main_conv    = nn.Conv2d(main_input_channels, dim, 7, padding=3)
        
        dims = [dim, *map(lambda m: dim*m, dim_mults)]

        # time embedding (match UNet)
        if with_time_emb:
            self.time_dim = dim
        else:
            self.time_dim = None

        self.downs      = nn.ModuleList([])
        self.zero_convs = nn.ModuleList([])
        
        # ---- DOWN ---- #
        num_res = len(dims) - 1
        ch      = dims[0]

        for i in range(num_res):
            next_ch = dims[i+1]
            
            self.downs.append(nn.ModuleList([
                self.block_klass(ch, ch, time_emb_dim = self.time_dim),
                self.block_klass(ch, ch, time_emb_dim = self.time_dim),
                Downsample(ch, next_ch) if i < (num_res - 1) else nn.Conv2d(ch, next_ch, 3, padding = 1)
            ]))
            
            self.zero_convs.append(nn.ModuleList([ZeroConv2d(ch, ch), ZeroConv2d(ch, ch)]))
            ch = next_ch 
 
        # ---- MID ---- #
        mid_dim = ch
        
        self.mid_block1 = self.block_klass(mid_dim, mid_dim, time_emb_dim=self.time_dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, LinearCrossAttention(mid_dim)))
        self.mid_block2 = self.block_klass(mid_dim, mid_dim, time_emb_dim=self.time_dim)
        
        self.mid_proj1  = ZeroConv2d(mid_dim, mid_dim)
        self.mid_proj2  = ZeroConv2d(mid_dim, mid_dim)


    def forward(self, main_input, control, t):
        """
        Returns list of residuals in the order they should be added to UNet_Basic's forward.
        """
        
        ctrl = self.init_control_conv(control)
        x    = self.init_main_conv(main_input)
        x    = x + ctrl
                
        residuals = []

        # ---- DOWN ---- #
        for (conv1, conv2, downsample), zero_conv in zip(self.downs, self.zero_convs):
            x = conv1(x, t);    residuals.append(zero_conv[0](x))
            x = conv2(x, t);    residuals.append(zero_conv[1](x))
            x = downsample(x)

        # ---- MID ---- #
        x = self.mid_block1(x, t); residuals.append(self.mid_proj1(x))
        x = self.mid_attn(x)
        x = self.mid_block2(x, t); residuals.append(self.mid_proj2(x))

        return residuals

