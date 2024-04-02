import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from timm.models.vision_transformer import Attention, Mlp
from pytorch3d.renderer import GridRaysampler, RayBundle, ray_bundle_to_ray_points

from utils.camera_utils import _get_camera_slice
from utils.ray_utils import DepthBasedMultinomialRaysampler
from utils.common_utils import HarmonicEmbedding, unnormalize
from mvdfusion.embedder import TimestepEmbedder, RayEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ModulatedLinearBlock(nn.Module):
    '''
    Time Conditioned Linear Block
    '''
    def __init__(self, input_dim, output_dim, cond_dim):
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=input_dim, hidden_features=output_dim, act_layer=approx_gelu, drop=0)
        self.input_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2*input_dim, bias=True)
        )
        self.output_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 1*output_dim, bias=True)
        )

    def forward(self, x, c):
        shift_mlp, scale_mlp = self.input_modulation(c).chunk(2, dim=1)
        gate_mlp = self.output_modulation(c)
        x = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTBlock(nn.Module):
    '''
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    '''
    def __init__(self, hidden_size, num_heads, cond_dim=None, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        if cond_dim is None:
            cond_dim = hidden_size

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class AggregationTransformer(nn.Module):
    '''
    '''
    def __init__(self, hidden_size, num_layers=3, num_heads=8, mlp_ratio=2.0, use_t=False):
        super().__init__()
        self.use_t = use_t
        layer_list = []
        for li in range(num_layers):
            if use_t:
                layer_list.append(DiTBlock(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio))
            else:
                raise NotImplementedError
        self.layer_list = nn.ModuleList(layer_list)
        self.weight_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, c=None):
        if self.use_t:
            for layer in self.layer_list:
                x = layer(x, c)
        else:
            for layer in self.layer_list:
                x = layer(x)
        x_weight = self.weight_layer(x)
        return x, x_weight
    

class GridAttn(nn.Module):
    '''
    Unproject Grid Feature Sampler
    '''
    def __init__(self,
                 input_size=32,
                 in_channels=4,
                 hidden_size=256,
                 output_dim=768,
                 num_heads=8,
                 mlp_ratio=2.0,
                 num_layers=3,
                 side_length=32,
                 world_scale=0.6,
                 z_near_far_scale=0.8,
                 depth_scale=2.0,
                 depth_shift=0.5,
                 n_pts_per_ray=3,
                 use_t=True,
                 keep_top_k_views=False,
                 top_k=4,
                 device='cpu'
                ):
        super().__init__()
        
        #@ INIT CUBE STUFF
        self.input_size = input_size
        self.device = device
        self.world_scale = world_scale
        self.side_length = side_length
        self.half_cube = 1.0*self.world_scale / self.side_length
        self.full_cube = 2.0*self.world_scale / self.side_length
        self.side_area = self.side_length * self.side_length
        self.num_cubes = self.side_area * self.side_length
        self.cube_index = torch.arange(self.num_cubes, device=self.device)
        self.cube_index_map = None
        self.cube_xyz = self._get_cube_center(self.cube_index).unsqueeze(0)
        self.z_near_far_scale = z_near_far_scale
        self.depth_scale = depth_scale
        self.depth_shift = depth_shift
        self.n_pts_per_ray = n_pts_per_ray
        self.keep_top_k_views = keep_top_k_views
        self.top_k = top_k

        #@ DEFINE HARMONIC EMBEDDING
        n_harmonic = 7
        omega0 = 0.1
        self.harmonic_embedding = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic,
            omega_0=omega0
        )
        depth_dim = self.harmonic_embedding.get_output_dim(1)
        plucker_dim = self.harmonic_embedding.get_output_dim(6)

        #@ INIT EMBEDDERS
        z_output_dim = 256
        self.z_embedder = nn.Sequential(nn.Linear(in_channels, z_output_dim), nn.GELU())
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.ray_embedder = RayEmbedder(input_size)

        #@ INIT LAYERS
        self.use_t = use_t
        self.pre_layer_b = nn.Sequential(nn.Linear(z_output_dim * 2 + plucker_dim * 2 + depth_dim * 2 + 1, hidden_size), nn.GELU())
        self.aggregation_transformer = AggregationTransformer(hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, mlp_ratio=mlp_ratio, use_t=use_t)

        ray_render_channels = hidden_size + depth_dim + plucker_dim
        self.final_layer_b = nn.Linear(hidden_size, output_dim)

        #@ INIT WEIGHTS
        self._initialize_weights()

    def _update_device(self, device):
        if self.cube_xyz.device != device:
            self.cube_xyz = self.cube_xyz.to(device)

    def _initialize_weights(self):

        #@ ZERO OUT MODULATION LAYERS
        for block in self.aggregation_transformer.layer_list:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _get_cube_center(self, cube_index):
        """
        Returns the coordinates of a given cube volume center in world [-1,1] space
        Args:
            cube_index: a Tensor of integer indices (...)
        Returns:
            xyz: coordinates in world space (..., 3)
        """
        if self.cube_index_map is None:
            #@ DEFINE NEW CUBE INDEX MAP
            cube_index = cube_index.float() # conver to float to support trunc
            xy_index = torch.trunc(cube_index%self.side_area)
            x_index = torch.trunc(xy_index%self.side_length)
            y_index = torch.trunc(xy_index/self.side_length)
            z_index  = torch.trunc(cube_index/(self.side_area))
            xyz = torch.stack((self.half_cube+x_index*2*self.half_cube-self.side_length*self.half_cube,
                            self.half_cube+y_index*2*self.half_cube-self.side_length*self.half_cube,
                            self.half_cube+z_index*2*self.half_cube-self.side_length*self.half_cube),
                            dim=-1)

            self.cube_index_map = xyz

        #@ MAP CUBE INDEX TO CUBE CENTER
        origin_shape = cube_index.shape
        xyz = self.cube_index_map[cube_index.flatten().long()].reshape((*origin_shape, 3))

        # negate xyz to correct for pytorch space 3d space
        return xyz
    
    def _encode_plucker(self, ray_origins, ray_dirs):
        """
        ray to plucker w/ pos encoding
        """
        plucker = torch.cat((ray_dirs, torch.cross(ray_origins, ray_dirs, dim=-1)), dim=-1)
        plucker = self.harmonic_embedding(plucker)
        return plucker 

    def index_features(self, input_feat, input_cameras, xyz, query_plucker=None, predict_mask=None):
        '''
        Returns features for each input xyz
        '''
        # print(xyz.shape)
        # exit()
        # assert len(xyz.shape) == 4
        # _, hw, d, _ = xyz.shape
        # xyz = rearrange(xyz, 'b n d c -> b (n d) c')

        xyz_cam = input_cameras.transform_points_ndc(xyz)
        # xyz_cam (NUM_INPUT_CAM, B*N, 3)
        
        xy = xyz_cam[...,:2].unsqueeze(2)
        # xy (NUM_INPUT_CAM, B*N, 1, 2)

        #@ GET CNN LATENT
        reference_features = torch.nn.functional.grid_sample(
                input_feat, # (B, L, H, W)
                -xy, # (B, N, 1, 2)
                align_corners=True,
                mode='bilinear',
                padding_mode='border',
            )
        
        reference_features = reference_features[...,0]
        reference_features = reference_features.permute(0,2,1)
        # reference_features (NUM_INPUT_CAM, N, FEAT_SIZE)

        #@ GET REFERENCE RAY
        origins_cam = rearrange(input_cameras.get_camera_center(), 'b c -> b () c')
        reference_dir = xyz.expand((len(input_cameras),-1,-1)) - origins_cam
        reference_depth = torch.linalg.norm(reference_dir, dim=-1, keepdim=True)
        reference_depth = self.harmonic_embedding(reference_depth)

        reference_dir = torch.nn.functional.normalize(reference_dir, dim=-1)
        reference_plucker = self._encode_plucker(origins_cam, reference_dir)
        
        #@ CONCATENATE FEATURES, DEPTH, AND DIR
        z = torch.cat((reference_features, reference_plucker, reference_depth),dim=-1)
        #' z (V B C)

        if predict_mask is not None:
            predict_mask_features = rearrange(predict_mask, 'v -> v () ()').expand(-1, z.shape[1], -1)
            z = torch.cat((z, predict_mask_features), dim=-1)

        if query_plucker is not None:
            query_plucker = rearrange(query_plucker, 'n d c -> () (n d) c').expand(len(input_cameras), -1, -1)
            z = torch.cat((z, query_plucker), dim=-1)

        z = rearrange(z, 'v b c -> b v c')
        return z
    
    
    def aggregate_features(self, cameras, depth_channel, t_embed=None, input_feat=None, input_latents=None, input_cameras=None, predict_mask=None):
        '''
        
        '''
        B = len(cameras)

        cam_dist_mean = torch.mean(torch.linalg.norm(cameras.get_camera_center(), axis=1))
        z_near = cam_dist_mean - self.z_near_far_scale
        z_far = cam_dist_mean + self.z_near_far_scale

        #@ DEFINE RAY SAMPLER
        half_pix = 1.0 / float(self.input_size)
        raysampler_grid = DepthBasedMultinomialRaysampler(
            min_x = 1.0 - half_pix,
            max_x = -1.0 + half_pix,
            min_y = 1.0 - half_pix,
            max_y = -1.0 + half_pix,
            image_height = self.input_size,
            image_width = self.input_size,
            n_pts_per_ray = 1,
        )

        #@ SHOOT RAYS ON DEPTHS
        ray_bundle = raysampler_grid(cameras, depth_channel)
        xyz_world = ray_bundle_to_ray_points(ray_bundle)
        #' xyz_world (B, H, W, 1, 3)
        #' origins (B, H, W, 3)
        #' directions (B, H, W, 3)
        #' lengths (B, H, W, N)

        #@ SAMPLE FEATURES ON RAYS
        b_, h_, w_, n_pts_per_ray_, _ = xyz_world.shape

        xyz_ = rearrange(xyz_world, 'b h w d c -> () (b h w d) c')
        xyz_cam = cameras.transform_points_ndc(xyz_)
        # xyz_cam (NUM_INPUT_CAM, B*N, 3)

        xy = xyz_cam[...,:2].unsqueeze(2)
        # xy (NUM_INPUT_CAM, B*N, 1, 2)

        #@ GET LATENT
        reference_features = torch.nn.functional.grid_sample(
                input_feat, # (B, L, H, W)
                -xy, # (B, N, 1, 2)
                align_corners=True,
                mode='bilinear',
                padding_mode='border',
            )
        #' (NUM_CAMERAS, C, B*H*W, 1)
        reference_features = rearrange(reference_features[...,0], 'v c (b h w d) -> v b (h w d) c', b=b_, h=h_, w=w_, d=n_pts_per_ray_)

        #@ GET INPUT LATENT
        xyz_cam_input = input_cameras.transform_points_ndc(xyz_)
        xy_input = xyz_cam_input[...,:2].unsqueeze(2)
        input_features = torch.nn.functional.grid_sample(
                input_latents, # (B, L, H, W)
                -xy_input, # (B, N, 1, 2)
                align_corners=True,
                mode='bilinear',
                padding_mode='border',
            )
        input_features = rearrange(input_features[...,0], 'v c (b h w d) -> v b (h w d) c', b=b_, h=h_, w=w_, d=n_pts_per_ray_)
        input_features = input_features.expand((len(cameras),-1,-1,-1))

        #@ GET REFERENCE RAY
        origins_cam = rearrange(cameras.get_camera_center(), 'b c -> b () c')
        reference_dir = xyz_.expand((len(cameras),-1,-1)) - origins_cam
        reference_dir = rearrange(reference_dir, 'v (b h w d) c -> v b (h w d) c', b=b_, h=h_, w=w_, d=n_pts_per_ray_)
        reference_depth = torch.linalg.norm(reference_dir, dim=-1, keepdim=True)
        reference_depth = self.harmonic_embedding(reference_depth)
        #' (V, BHWD, C)

        reference_dir = torch.nn.functional.normalize(reference_dir, dim=-1)
        reference_plucker = self._encode_plucker(origins_cam.unsqueeze(1), reference_dir)
        #' (V, BHWD, C)

        #@ GET QUERY RAY
        query_dir = ray_bundle.directions
        query_dir = torch.nn.functional.normalize(query_dir, dim=-1)
        query_dir = repeat(query_dir, 'b h w c -> () b (h w d) c', h=h_, w=w_, d=n_pts_per_ray_)

        query_origin = cameras.get_camera_center()
        query_origin = repeat(query_origin, 'b c -> b h w d c', h=h_, w=w_, d=n_pts_per_ray_)
        query_origin = rearrange(query_origin, 'b h w d c -> () b (h w d) c')
        query_plucker = self._encode_plucker(query_origin, query_dir)
        query_plucker = query_plucker.expand((len(cameras), -1, -1, -1))
        #' all shapes (V, BHWD, C)

        #@ GET QUERY RAY EMBED
        query_depths = rearrange(ray_bundle.lengths.unsqueeze(-1), 'b h w d c -> () b (h w d) c')
        query_depths = self.harmonic_embedding(query_depths)
        query_depths = query_depths.expand((len(cameras), -1, -1, -1))
        #' all shapes (V, BHWD, C)
        
        #@ CONCATENATE FEATURES, DEPTH, AND DIR
        # print(reference_features.shape, input_features.shape, reference_plucker.shape, reference_depth.shape, query_plucker.shape)
        z = torch.cat((reference_features, input_features, reference_plucker, reference_depth, query_plucker, query_depths),dim=-1)
        #' z (V B HWD C)

        if predict_mask is not None:
            predict_mask_features = rearrange(predict_mask, 'v -> v () () ()').expand(-1, z.shape[1], z.shape[2], -1)
            z = torch.cat((z, predict_mask_features), dim=-1)

        #! ###
        #@ TOP K
        #! ###
        if self.keep_top_k_views:
            top_k_row = torch.arange(-(self.top_k//2), self.top_k//2+1)
            top_k_row = top_k_row.unsqueeze(1).expand(-1, len(reference_features))
            top_k_idx_offset = torch.arange(len(reference_features)).unsqueeze(0)
            top_k_row = top_k_row + top_k_idx_offset
            top_k_idx = top_k_row % len(reference_features)
            top_k_idx = top_k_idx.to(z.device)

            top_k_idx = top_k_idx[:,:,None,None].expand(-1, -1, z.shape[2], z.shape[3])
            z = torch.gather(z, 0, top_k_idx)

        v__, b__, _, _ = z.shape
        z = rearrange(z, 'v b n c -> v (b n) c')

        #@ AGGREGATE ACROSS V
        aggregation_input = rearrange(z, 'v b c -> b v c')
        aggregation_input = self.pre_layer_b(aggregation_input)
        if self.use_t:
            output, weights = self.aggregation_transformer(aggregation_input, t_embed)
        else:
            output, weights = self.aggregation_transformer(aggregation_input)
        weights = torch.nn.functional.softmax(weights, dim=-2)
        aggregated_feat = (output * weights).sum(dim=-2)
        #' N = BHW
        #' output (N, C)

        #@ RESHAPE
        aggregated_feat = rearrange(aggregated_feat, '(b h w d) c -> (b h w) d c', b=b_, h=h_, w=w_, d=n_pts_per_ray_)

        #! POTENTIAL TRANSFORMER ACROSS D
        #! PORTENTIAL TRANSFORMER ACROSS HW
        feature_frustum = self.final_layer_b(aggregated_feat)
        feature_frustum = rearrange(feature_frustum, '(b h w) d c -> b h w d c', b=b_, h=h_, w=w_)
        #' (B, H, W, D, C)

        return feature_frustum
    
    
    def forward(self, noisy_latents, batch_cameras, predict_mask, t_embed, t, scheduler, overwrite_attn_depth=None, input_latents=None, input_cameras=None):

        self._update_device(noisy_latents.device)

        #@ GET UNBIASED DEPTH ESTIMATE
        if overwrite_attn_depth is None:
            depth_channel = noisy_latents[:,4:,...]
            sqrt_alphas_cumprod_  = scheduler.sqrt_alphas_cumprod[t]
            depth_std_ = scheduler.sqrt_one_minus_alphas_cumprod[t] / scheduler.sqrt_alphas_cumprod[t] / 10.0
            depth_channel = depth_channel / sqrt_alphas_cumprod_.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            depth_std_ = scheduler.sqrt_one_minus_alphas_cumprod[t] / scheduler.sqrt_alphas_cumprod[t] / 10.0
            depth_channel = overwrite_attn_depth

        #@ SAMPLE DEPTH PTS
        depth_channel = depth_channel.expand(-1, self.n_pts_per_ray, -1, -1)
        _, n_d, h_d, w_d = depth_channel.shape
        depth_std = rearrange(depth_std_, 'b -> b () () ()').expand(-1, n_d, h_d, w_d)
        depth_samples = torch.normal(depth_channel, std=depth_std)
        depth_samples = unnormalize(depth_samples) * self.depth_scale + self.depth_shift

        input_feat = self.z_embedder(rearrange(noisy_latents, 'b c h w -> b h w c'))
        input_feat = rearrange(input_feat, 'b h w c -> b c h w')
        input_latents = self.z_embedder(rearrange(input_latents, 'b c h w -> b h w c'))
        input_latents = rearrange(input_latents, 'b h w c -> b c h w')
        t_embed = t_embed[:1]
        
        assert noisy_latents.shape[1] == 5, 'depth wise efficient attention requires 4+1 channels'
        feature_frustum = self.aggregate_features(batch_cameras, depth_samples, t_embed=t_embed, input_feat=input_feat, predict_mask=predict_mask, input_latents=input_latents, input_cameras=input_cameras)
        return feature_frustum