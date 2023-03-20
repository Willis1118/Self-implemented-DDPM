import math
from functools import partial

import torch
from torch import nn, einsum

from tqdm.auto import tqdm
from einops import rearrange

from helper import *

'''
    Below we define the Convolutional Module needed by U-Net
'''

class Residual(nn.Module):
    '''
        define a class of residual connection
    '''
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        # residual connection
        # takes in iterable and dict
        return self.fn(x, *args, **kwargs) + x

class PositionalEmbeddings(nn.Module):
    '''
        define the positional embedding for time step t
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        '''
            sin cos embedding; refer to attention is all you need
            input t is of shape (batch, 1), and output embeddings is of shape (batch, dim), 
            where dim is the dimensionality of positional embeddings.
        '''
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(1e4) / (half_dim - 1)

        #create a tensor [0, half_dim)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) 

        # (batch, 1) * (1, dim)
        embeddings = t[:, None] * embeddings[None, :] 

        # cat 2*i and 2*i+1 together
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    '''
        A single block in U-Net structure defined in DDPM
    '''

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)

        # input channels to be divide into groups, each group has num_channels / num_groups channels
        # each group is then normalized within itself
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))

        # scale x by positional embeddings
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        return self.act(x)

class ResnetBlock(nn.Module):
    '''
        Original structure proposed by DDPM
    '''
    # * here refers to the end of positional arguments; every argument after * must be specified with keyword
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        # time embeddings projection
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out)
            )
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        # in case dim not equal to dim_out, to reconcilize the dimension of input and output
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        # x should be of shape (b c h w)
        h = self.block1(x)

        # additive time embeddings
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, 'b c -> b c 1 1') + h
        
        h = self.block2(h)
        return h + self.res_conv(x)

class ConvNextBlock(nn.Module):
    '''
        new structure proposed by Phil Wang, which improves on performance
    '''
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, dim_out)
            )
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_emb=None):
        '''
            ConvNext Block structure
        '''

        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')
        
        h = self.net(h)
        return h + self.res_conv(x)

'''
    Below we defined the Attention Module needed by U-Net
'''

class Attention(nn.Module):
    '''
        The multiplicative multiheaded attention
    '''
    def __init__(self, dim, heads=4, dim_head=32):
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = heads * dim_head

        # linear projection for every pixel
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # after calculated QK^T V
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # split channel of hidden_dim * 3 to hidden_dim
        qkv = self.to_qkv(x).chunk(3, dim=1) 

        # linearly project pixels from (x, y) to x * y
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )

        q = q * self.scale

        # it does not matter if we transpose K here; it can be viewed as we generate K^T from the beginning
        sim = einsum('b h d i, b h d j -> b h i j', q, k)

        # normalize the product
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # rearrange the output to image form
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    '''
        The linear projected attention using mlp
    '''
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )

        # softmax first
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1) #--> transpose

        q = q * self.scale
        context = einsum('b h d n, b h e n -> b h d e', k, v)

        out = einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


'''
    The author of DDPM interleaves the conv / attn layer with group norm
    below we define the PreNorm class, which is solely for applying group norm before attention layer
'''

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    
    def forward(self, x):
        return self.fn(self.norm(x))

'''
    Below we defined the conditional U-Net:
    By nature it is to predict the noise added to noisy image x_t at time t, and output the noise added to x_t
'''

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1,2,4,8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2
    ):
        super().__init__()

        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)

        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims=[init_dim, *map(lambda m: dim*m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_class = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_class = partial(ResnetBlock, groups=resnet_block_groups)
        
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                PositionalEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None
        
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # down sampling part of U-Net
        # by default will have 4 layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    block_class(dim_in, dim_out, time_emb_dim=time_dim),
                    block_class(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ])
            )
        
        # mid connection after finishing the down sampling
        mid_dim = dims[-1] # last dimension --> dim * 8
        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= num_resolutions - 1

            self.ups.append(
                nn.ModuleList([
                    block_class(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    block_class(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                ])
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_class(dim, dim), 
            nn.Conv2d(dim, out_dim, 1)
        )
    
    def forward(self, x, time):

        x = self.init_conv(x)

        # Positional Encoding
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # residual across downs and ups
        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)

            h.append(x)
            x = downsample(x)
        
        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) # cat on channel level
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        
        # final layer
        return self.final_conv(x)