import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange

from pytorch3d.renderer import PerspectiveCameras, GridRaysampler

from utils.common_utils import HarmonicEmbedding


class RayEmbedder(nn.Module):
    '''
    
    '''
    def __init__(self, input_size, n_harmonic=7, omega0=0.1):
        '''
        Args
            input_size (h) - the side length of patches per image
        '''
        super().__init__()
        self.input_size = input_size

        #@ DEFINE HARMONIC EMBEDDING
        self.harmonic_embedding = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic,
            omega_0=omega0
        )
        self.output_dim = self.harmonic_embedding.get_output_dim(6)

        #@ DEFINE RAY SAMPLER
        img_h, img_w = input_size, input_size
        half_pix_width = 1.0 / img_w
        half_pix_height = 1.0 / img_h
        self.raysampler_grid = GridRaysampler(
            min_x=1.0-half_pix_height,
            max_x=-1.0+half_pix_height,
            min_y=1.0-half_pix_width,
            max_y=-1.0+half_pix_width,
            image_height=img_h,
            image_width=img_w,
            n_pts_per_ray=2,
            min_depth=0.8,
            max_depth=1.8
        )

    def forward(self, cameras, return_plucker=False):

        ray_bundle = self.raysampler_grid(cameras)

        ray_origins = ray_bundle.origins
        #' ray_origins (B, H, W, 3)

        ray_dirs = torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        #' ray_dirs (B, H, W, 3)

        plucker = torch.cat((ray_dirs, torch.cross(ray_origins, ray_dirs, dim=-1)), dim=-1)
        #' plucker (B, H, W, 6)

        plucker_embed = self.harmonic_embedding(plucker)
        #' plucker_embed (B, H, W, F)

        plucker_embed = rearrange(plucker_embed, 'b h w f -> b f h w')

        if return_plucker:
            plucker = rearrange(plucker, 'b h w f -> b f h w')
            return plucker_embed, plucker
        
        return plucker_embed
    

#@ DIT Time Embedder
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#@ Stable Diffusion Time Embedder
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding