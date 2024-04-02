from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from external.sd1.ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from external.sd1.ldm.modules.diffusionmodules.openaimodel import (
    SpatialTransformer,
    TimestepBlock,
    ResBlock,
    Downsample,
    Upsample,
    convert_module_to_f16,
    convert_module_to_f32
)
from mvdfusion.attention import ViewAlignedFeatureTransformer
from utils.load_model import instantiate_from_config, load_model_from_config

'''
Sequential Wrapper Class
'''
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, volume_feat=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, ViewAlignedFeatureTransformer):
                x = layer(x, volume_feat)
            else:
                x = layer(x)
        return x


#@ from https://github.com/liuyuan-pal/SyncDreamer/
class UNetWrapper(nn.Module):
    def __init__(self, 
                 unet_config, 
                 unet_path=None, 
                 drop_conditions=False, 
                 drop_scheme='default', 
                 use_zero_123=False, 
                 finetune_unet=False,
                 finetune_cross_attn=False,
                 finetune_view_attn=True,
                 remove_keys=[]):
        super().__init__()

        #@ FIXED PARAM TO MAP FROM ORIGINAL LAYERS TO MODIFIED POSITIONS
        param_mapper = {
            'output_blocks.5.2.conv.weight':'output_blocks.5.3.conv.weight',
            'output_blocks.5.2.conv.bias':'output_blocks.5.3.conv.bias',
            'output_blocks.8.2.conv.weight':'output_blocks.8.3.conv.weight',
            'output_blocks.8.2.conv.bias':'output_blocks.8.3.conv.bias',
            'middle_block.2.in_layers.0.weight':'middle_block.3.in_layers.0.weight',
            'middle_block.2.in_layers.0.bias':'middle_block.3.in_layers.0.bias',
            'middle_block.2.in_layers.2.weight':'middle_block.3.in_layers.2.weight',
            'middle_block.2.in_layers.2.bias':'middle_block.3.in_layers.2.bias',
            'middle_block.2.emb_layers.1.weight':'middle_block.3.emb_layers.1.weight',
            'middle_block.2.emb_layers.1.bias':'middle_block.3.emb_layers.1.bias',
            'middle_block.2.out_layers.0.weight':'middle_block.3.out_layers.0.weight',
            'middle_block.2.out_layers.0.bias':'middle_block.3.out_layers.0.bias',
            'middle_block.2.out_layers.3.weight':'middle_block.3.out_layers.3.weight',
            'middle_block.2.out_layers.3.bias':'middle_block.3.out_layers.3.bias'

        }
        
        self.unet_model = load_model_from_config(unet_config, unet_path, verbose=False,
                                                 replace_key=['model.diffusion_model.',''],
                                                 ignore_keys=['aligned_attn_'],
                                                 param_mapper=param_mapper,
                                                 remove_keys=remove_keys,
                                                 )

        if not finetune_unet:
            self.unet_model.disable_unet_grad()

        self.drop_conditions = drop_conditions
        self.drop_scheme=drop_scheme
        self.use_zero_123 = use_zero_123
        self.finetune_unet = finetune_unet
        self.finetune_cross_attn = finetune_cross_attn
        self.finetune_view_attn = finetune_view_attn
        print('*drop conditions: ', self.drop_conditions)

    def drop(self, cond, mask):
        shape = cond.shape
        B = shape[0]
        cond = mask.view(B,*[1 for _ in range(len(shape)-1)]) * cond
        return cond

    def get_trainable_parameters(self):
        if self.finetune_unet:
            return self.unet_model.parameters()
        else:
            return self.unet_model.get_cross_attn_parameters(finetune_cross_attn=self.finetune_cross_attn, finetune_view_attn=self.finetune_view_attn)

    def get_drop_scheme(self, B, device):
        if self.drop_scheme=='default':
            random = torch.rand(B, dtype=torch.float32, device=device)
            drop_clip = (random > 0.15) & (random <= 0.2)
            drop_volume = (random > 0.1) & (random <= 0.15)
            drop_concat = (random > 0.05) & (random <= 0.1)
            drop_all = random <= 0.05
        else:
            raise NotImplementedError
        return drop_clip, drop_volume, drop_concat, drop_all

    def forward(self, x, t, clip_embed, volume_feats, x_concat=None, is_train=False):
        """

        @param x:             B,4,H,W
        @param t:             B,
        @param clip_embed:    B,M,768
        @param volume_feats:  B,C,D,H,W
        @param x_concat:      B,C,H,W
        @param is_train:
        @return:
        """
        if self.drop_conditions and is_train:
            B = x.shape[0]
            drop_clip, drop_volume, drop_concat, drop_all = self.get_drop_scheme(B, x.device)

            clip_mask = 1.0 - (drop_clip | drop_all).float()
            clip_embed = self.drop(clip_embed, clip_mask)

            volume_mask = 1.0 - (drop_volume | drop_all).float()
            volume_feats = self.drop(volume_feats, mask=volume_mask)

            concat_mask = 1.0 - (drop_concat | drop_all).float()
            x_concat = self.drop(x_concat, concat_mask)

        if self.use_zero_123 and x_concat is not None:
            # zero123 does not multiply this when encoding, maybe a bug for zero123
            first_stage_scale_factor = 0.18215
            x_concat_ = x_concat * 1.0
            x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
        else:
            x_concat_ = x_concat
        if x_concat is not None:
            x = torch.cat([x, x_concat_], 1)
        volume_levels = self.get_volume_feats_pyramid(volume_feats)
        pred = self.unet_model(x, t, clip_embed, volume_feats=volume_levels)
        return pred

    @torch.no_grad()
    def predict_with_unconditional_scale(self, x, t, clip_embed, volume_feats, x_concat, unconditional_scale):
        # x_ = torch.cat([x] * 2, 0)
        # t_ = torch.cat([t] * 2, 0)
        t_ = t

        clip_embed_ = clip_embed.clone()
        clip_embed_null = torch.zeros_like(clip_embed)
        # clip_embed_null = clip_embed.clone()

        x_concat_ = x_concat.clone()
        x_concat_null = torch.zeros_like(x_concat)
        # x_concat_null = x_concat.clone()

        if self.use_zero_123:
            # zero123 does not multiply this when encoding, maybe a bug for zero123
            first_stage_scale_factor = 0.18215
            x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
            x_concat_null[:, :4] = x_concat_null[:, :4] / first_stage_scale_factor

        x_ = torch.cat([x, x_concat_], 1)
        x_null = torch.cat([x, x_concat_null], 1)

        volume_levels = self.get_volume_feats_pyramid(volume_feats)
        volume_levels_null = self.get_volume_feats_pyramid(torch.zeros_like(volume_feats))

        s = self.unet_model(x_, t_, clip_embed_, volume_feats=volume_levels)
        s_uc = self.unet_model(x_null, t_, clip_embed_null, volume_feats=volume_levels_null)
        
        s = s_uc + unconditional_scale * (s - s_uc)
        return s

    def get_volume_feats_pyramid(self, volume_feats):
        b, h, w, d, c = volume_feats.shape
        volume_feats = rearrange(volume_feats, 'b h w d c -> (b d) c h w')
        feats_list = []
        num_levels = len(self.unet_model.channel_mult)
        level_map = {0:1.0, 1:1.0, 2:0.5,3:0.25}
        for i in range(num_levels):
            scale_i = 0.5 ** i
            layer_feat = torch.nn.functional.interpolate(volume_feats, scale_factor=scale_i, mode='area')
            layer_feat = rearrange(layer_feat, '(b d) c h w -> b h w d c', b=b, d=d)
            feats_list.append(layer_feat)
        return feats_list


'''
View Conditioned UNet
'''
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,    # custom transformer support
        use_view_aligned_transformer=True,
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ):
        super().__init__()

        assert use_view_aligned_transformer, 'This module only supports spatial view aligned transformer!'
        assert use_spatial_transformer, 'This module only supports spatial view aligned transformer'
        assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        from omegaconf.listconfig import ListConfig
        if type(context_dim) == ListConfig:
            context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        #@ INPUT BLOCKS
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
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
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        #@ MIDDLE BLOCKS
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            #! #################################
            #! BEGIN MODIFIED FROM ORIGINAL UNET
            #! #################################
            #@ ADD ADDITIONAL CROSS ATTENTION LAYER
            ViewAlignedFeatureTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, image_size=image_size
            ),
            #! #################################
            #! END MODIFIED FROM ORIGINAL UNET
            #! #################################

            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        #@ OUTPUT BLOCKS
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                    
                    #! #################################
                    #! BEGIN MODIFIED FROM ORIGINAL UNET
                    #! #################################
                    #@ ADD ADDITIONAL CUSTOM ATTENTION LAYER
                    layers.append(
                        ViewAlignedFeatureTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, image_size=image_size
                        )
                    )
                    #! #################################
                    #! END MODIFIED FROM ORIGINAL UNET
                    #! #################################
                    
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
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

    def forward(self, x, timesteps=None, context=None, y=None, volume_feats=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context, volume_feats)
            hs.append(h)
        h = self.middle_block(h, emb, context, volume_feats)
        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, volume_feats)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
        
    def get_cross_attn_parameters(self, finetune_cross_attn, finetune_view_attn):

        param_list = []
        for name, param in self.named_parameters():
            # print(name, f'{torch.prod(torch.tensor(param.shape)) * 1.e-3:.2f}K')
            if finetune_cross_attn:
                if '.norm.' in name or '.proj_in.' in name or '.transformer_blocks.' in name or '.proj_out.' in name:
                    param_list.append(param)
                
            if finetune_view_attn:
                if '.aligned_attn_' in name:
                    param_list.append(param)
        return param_list

    def disable_unet_grad(self):

        for name, param in self.named_parameters():
            if '.aligned_attn_' not in name:
                param.requires_grad_(False)