from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from external.sd1.ldm.modules.diffusionmodules.util import checkpoint

from external.sd1.ldm.modules.attention import exists, einsum, Normalize, zero_module, CrossAttention, FeedForward

    
'''
Dual Attention Block
'''
class DualAttnetionBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, preserve_unet_dim=False):
        super().__init__()

        #@ INIT
        attn_mode = "softmax"
        assert attn_mode in self.ATTENTION_MODES
        assert disable_self_attn is False
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn

        #@ SET UP LAYERS
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=None) 
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=context_dim if not preserve_unet_dim else None)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, shape=None):
        return checkpoint(self._forward, (x, context, shape), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, shape=None):
        '''
        Args:
            x (B, HW, C) - 
            context (B, HW, D, C) - 
        '''
        b, c, h, w = shape

        #@ SELF ATTENTION
        x = self.attn1(self.norm1(x), context=None) + x
        #' x (B, HW, F)
        
        #@ CROSS ATTENTION
        x = rearrange(x, 'b (h w) c -> (b h w) () c', b=b, h=h, w=w)
        context = rearrange(context, 'b (h w) d c -> (b h w) d c', b=b, h=h, w=w)
        #' x (BHW, 1, C)
        #' context (BHW, D, C)
        
        x = self.attn2(self.norm2(x), context=context) + x
        x = rearrange(x[:, 0, ...], '(b h w) c -> b (h w) c', b=b, h=h, w=w)

        #@ LINEAR
        x = self.ff(self.norm3(x)) + x
        return x


'''
View Aligned Transformer Module
'''
class ViewAlignedFeatureTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=True,
                 use_checkpoint=True,
                 image_size=None):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.image_size = image_size
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.aligned_attn_norm = Normalize(in_channels)
        if not use_linear:
            self.aligned_attn_proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.aligned_attn_proj_in = nn.Linear(in_channels, inner_dim)

        self.aligned_attn_transformer_blocks = nn.ModuleList(
            [DualAttnetionBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.aligned_attn_proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.aligned_attn_proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

        self.level_mapper = {self.image_size:0, self.image_size//2:1, self.image_size//4:2, self.image_size//8:3}

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        #' x (B, C, H, W)
        #' context [(B, H, W, D, C),...]

        shape = x.shape
        b, c, h, w = x.shape

        context_level = self.level_mapper[h]
        context = rearrange(context[context_level], 'b h w d c -> b (h w) d c')
        #' context (B, HW, D, C)
        
        x_in = x
        x = self.aligned_attn_norm(x)
        if not self.use_linear:
            x = self.aligned_attn_proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.aligned_attn_proj_in(x)
        for i, block in enumerate(self.aligned_attn_transformer_blocks):
            x = block(x, context=context, shape = shape)
        if self.use_linear:
            x = self.aligned_attn_proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.aligned_attn_proj_out(x)
        return x + x_in
    

'''
View Aligned Transformer Module
'''
class MultiviewEpipolarTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=True,
                 use_checkpoint=True,
                 image_size=None):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.image_size = image_size
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.aligned_attn_norm = Normalize(in_channels)
        if not use_linear:
            self.aligned_attn_proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.aligned_attn_proj_in = nn.Linear(in_channels, inner_dim)

        self.aligned_attn_transformer_blocks = nn.ModuleList(
            [DualAttnetionBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, preserve_unet_dim=True)
                for d in range(depth)]
        )
        if not use_linear:
            self.aligned_attn_proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.aligned_attn_proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

        self.level_mapper = {self.image_size:0, self.image_size//2:1, self.image_size//4:2, self.image_size//8:3}

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        #' x (B, C, H, W)
        #' context [(B, H, W, D, C),...]

        shape = x.shape
        b_, c, h_, w_ = x.shape
        n_pts_per_ray_ = 1

        context_level = self.level_mapper[h_]
        uv = context[context_level]
        #' x (B, C, H, W)
        #' uv (V, BHW, 1, 2)

        #@ SAMPLE FEATURES
        #' reference_features_i = grid_sample(x, uv[i].view(B, HW, 1, 2))
        reference_features = torch.nn.functional.grid_sample(
            x,
            -uv, #' uv = rearrange(uv, 'v (bhw) a b -> b (vhw) a b')
            align_corners=True,
            mode='bilinear',
            padding_mode='border',
        )
        reference_features = rearrange(reference_features[...,0], 'v c (b h w d) -> b (h w) (v d) c', b=b_, h=h_, w=w_, d=n_pts_per_ray_)
        #' reference_features (B, HW, BD, C)
        
        x_in = x
        x = self.aligned_attn_norm(x)
        if not self.use_linear:
            x = self.aligned_attn_proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.aligned_attn_proj_in(x)
        for i, block in enumerate(self.aligned_attn_transformer_blocks):
            x = block(x, context=reference_features, shape = shape)
        if self.use_linear:
            x = self.aligned_attn_proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h_, w=w_).contiguous()
        if not self.use_linear:
            x = self.aligned_attn_proj_out(x)
        return x + x_in