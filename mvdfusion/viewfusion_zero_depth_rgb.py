import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat

from pytorch3d.renderer import PerspectiveCameras

from external.sd1.ldm.modules.encoders.modules import FrozenCLIPImageEmbedder

from mvdfusion.scheduler import DDPMScheduler
from mvdfusion.unet import UNetWrapper
from mvdfusion.embedder import timestep_embedding
from utils.common_utils import normalize, unnormalize, split_list, disable_training_module
from utils.camera_utils import _get_camera_slice, _get_relative_camera
from utils.load_model import instantiate_from_config, load_model_from_config
from mvdfusion.sampler import DDIMSampler

class ViewFusion(nn.Module):

    def __init__(self,
                view_attn_config,
                unet_config,
                ddpm_config,
                vae_config,
                unet_path='',
                vae_path='',
                clip_path='',
                unet_cc_path='/scratch/zhizhuoz/sd_weights/zero123_105000_cc.ckpt',
                z_scale_factor=0.18215,
                vae_max_batch=8,
                objective='noise',
                loss_type='l2',
                embed_camera_pose=True,
                finetune_projection=False,
                finetune_unet=False,
                finetune_cross_attn=True,
                finetune_view_attn=True,
                feed_prev_depth=False,
                drop_conditions=False,
                **kwargs):
        super().__init__()
        self.finetune_projection = finetune_projection
        self.finetune_unet = finetune_unet
        self.z_scale_factor = z_scale_factor
        self.vae_max_batch = vae_max_batch
        self.objective = objective
        self.loss_type = loss_type
        self.embed_camera_pose = embed_camera_pose
        self.finetune_cross_attn = finetune_cross_attn
        self.finetune_view_attn = finetune_view_attn
        self.unet_path = unet_path
        self.unet_cc_path = unet_cc_path
        self.feed_prev_depth = feed_prev_depth
        self.drop_conditions = drop_conditions

        #@ INIT VIEW ATTN
        self.view_attn = instantiate_from_config(view_attn_config)

        #@ INIT UNET
        self.unet_model = UNetWrapper(unet_config, 
                                      unet_path=unet_path, 
                                      drop_conditions=drop_conditions, 
                                      drop_scheme='default',
                                      finetune_unet=finetune_unet, 
                                      finetune_cross_attn=finetune_cross_attn,
                                      finetune_view_attn=finetune_view_attn,
                                      use_zero_123=True,
                                      remove_keys=['input_blocks.0.0.weight', 'out.2.weight', 'out.2.bias'])  

        #@ INIT DDPM SCHEDULER
        self.scheduler = instantiate_from_config(ddpm_config)

        #@ INIT VAE
        self.vae = load_model_from_config(vae_config, vae_path, replace_key=['first_stage_model.',''], verbose=False)

        #@ INIT CLIP
        self._init_clip(clip_path)
        self._init_clip_projection()
        self._init_time_step_embedding()

        #@ FREE PARAM FOR DEVICE LOGGING
        self.register_buffer('_device', torch.tensor([0.]), persistent = False)

        #@ INIT LOSS
        if loss_type == 'l2':
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise NotImplementedError

        #@ INIT DDIM SAMPLER
        self.ddim = DDIMSampler(self, ddim_num_steps=50, ddim_discretize="uniform", ddim_eta=1.0, latent_size=32, z_dim=4, feed_prev_depth=feed_prev_depth)

        #@ ASSERT CHECKS
        assert self.finetune_view_attn is True, 'must finetune new view attention layers'

        if self.embed_camera_pose:
            assert self.finetune_projection is True, 'embedding camera'

        print('*using prev_depth_0', self.feed_prev_depth)

            
    def _init_clip(self, clip_path):
        self.clip_image_encoder = FrozenCLIPImageEmbedder(model=clip_path)
        self.clip_image_encoder = disable_training_module(self.clip_image_encoder)

    def _init_clip_projection(self):
        if self.embed_camera_pose:
            # self.cc_projection = nn.Linear(768 + 14*2, 768)
            self.cc_projection = nn.Sequential(nn.Linear(768 + 14*2, 768), nn.SiLU(True), nn.Linear(768, 768), nn.SiLU(True), nn.Linear(768, 768))
        else:
            self.cc_projection = nn.Linear(768+4, 768)
        nn.init.eye_(list(self.cc_projection.parameters())[0][:768, :768])
        nn.init.zeros_(list(self.cc_projection.parameters())[1])
        self.cc_projection.requires_grad_(True)

        if not self.embed_camera_pose:
            pl_sd = torch.load(self.unet_cc_path, map_location="cpu")
            m, u = self.load_state_dict(pl_sd['state_dict'], strict=False)
            assert len(u) == 0
            print('Loading cc_projection from', self.unet_cc_path)

        if not self.finetune_projection:
            disable_training_module(self.cc_projection)

    def _init_time_step_embedding(self):
        self.time_embed_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(True),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

    def _print_parameter_count(self):

        view_attn_p = sum(p.numel() for p in self.view_attn.parameters())
        unet_p = sum(p.numel() for p in self.unet_model.get_trainable_parameters())
        unet_full = sum(p.numel() for p in self.unet_model.parameters())
        time_p = sum(p.numel() for p in self.time_embed.parameters())
        projection_p = sum(p.numel() for p in self.cc_projection.parameters()) if self.finetune_projection else 0

        total_p = view_attn_p + unet_p + time_p + projection_p
        total_p2 = view_attn_p + unet_full + time_p + projection_p

        print('--------training params--------')
        print(f'view_attn params: {view_attn_p * 1.e-6:.2f}M')
        print(f'unet params: {unet_p * 1.e-6:.2f}M')
        print(f'total params: {total_p * 1.e-6:.2f}M')
        print('----------total params---------')
        print(f'unet full params: {unet_full * 1.e-6:.2f}M')
        print(f'total full params: {total_p2 * 1.e-6:.2f}M')

    @torch.no_grad()
    def encode_clip(self, x):
        return self.clip_image_encoder.encode(x)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(normalize(x)).mode() * self.z_scale_factor
    
    @torch.no_grad()
    def decode(self, z):
        return unnormalize(self.vae.decode(z * 1 / self.z_scale_factor)).clip(0.0, 1.0)
    
    def prepare_batch(self, batch, trainer_config, generator=None):
        '''
        Args:
            batch:                          - a batch from dataloader
                images (S, 3, H*, W*)
                masks  
                R (S, 3, 3)
                T (S, 3)
                f (S, 2)
                c (S, 2)
            trainer_config                  - config that specifies input size and render batch size
            generator                       - optional random generator

        Returns:
            noisy_latents (B, 4, H, W)      - noisy latents
            batch_cameras (B, )             - noisy pytorch3d cameras
            input_latents (1, 4, H, W)      - input image latentes
            input_cameras (1, )             - input pytorch3d camera
            clip_embed (1, 1, 768)          - clip feature for input image

        '''
        batch_device = batch['images'].device
        B, _, H, W = batch['images'].shape

        #@ SAMPLE IDX
        input_batch_size = trainer_config['input_batch_size']
        train_batch_size = trainer_config['train_batch_size']
        if trainer_config['random_views']:
            if generator is not None:
                rand = torch.randperm(B, generator=generator)
            else:
                rand = torch.randperm(B)
        else:
            rand = torch.linspace(0, B-1, input_batch_size+train_batch_size).long()
        input_idx = rand[:input_batch_size]
        batch_idx = rand[input_batch_size:train_batch_size+input_batch_size]

        #@ ENCODE IMAGES
        input_images = batch['images'][input_idx]
        input_latents = self.encode(batch['images'][input_idx])
        batch_latents = self.encode(batch['images'][batch_idx])

        #@ ADD DEPTHS
        if 'depths' in batch:
            input_depths = normalize(batch['depths'][input_idx]).to(input_latents.device)
        else:
            input_depths = torch.zeros((input_batch_size, 1, H, W), device=input_latents.device)
        input_depths = torch.nn.functional.interpolate(input_depths, scale_factor=0.125, mode='area')
        
        #! OVERWRITE INPUT DEPTH
        input_depths = torch.zeros_like(input_depths)
        input_latents = torch.cat((input_latents, input_depths), dim=1)
        if 'depths' in batch:
            batch_depths = normalize(batch['depths'][batch_idx]).to(input_latents.device)
        else:
            batch_depths = torch.zeros((train_batch_size, 1, H, W), device=input_latents.device)
        batch_depths = torch.nn.functional.interpolate(batch_depths, scale_factor=0.125, mode='area')
        batch_latents = torch.cat((batch_latents, batch_depths), dim=1)

        #' input_latents (1, C, H, W)
        #' batch_latents (B, C, H, W)

        #@ CONSTRUCT CAMERAS
        cameras = PerspectiveCameras(
                            R=batch['R'], 
                            T=batch['T'], 
                            focal_length=batch['f'], 
                            principal_point=batch['c'], 
                            # image_size=batch['image_size']
                        ).to(batch_device)
    
        #@ RELATIVE CAMERAS
        cameras = _get_relative_camera(cameras, query_idx=input_idx)
        input_cameras = _get_camera_slice(cameras, input_idx)
        batch_cameras = _get_camera_slice(cameras, batch_idx)
        
        #@ GET CLIP EMBED
        clip_embed = self.encode_clip(input_images)
        clip_embed = clip_embed.expand(train_batch_size, -1, -1)

        #@ EMBED CAMERA POSE PARAMETERS
        #' clip_embed (B, 1, 768)
        if self.embed_camera_pose:
            input_r = rearrange(input_cameras.R, 'b r c -> b () (r c)')
            input_t = rearrange(input_cameras.T, 'b c -> b () c')
            input_f = rearrange(input_cameras.focal_length, 'b c -> b () c')
            batch_r = rearrange(batch_cameras.R, 'b r c -> b () (r c)')
            batch_t = rearrange(batch_cameras.T, 'b c -> b () c')
            batch_f = rearrange(batch_cameras.focal_length, 'b c -> b () c')
            input_cam_embed = torch.cat((input_r, input_t, input_f), dim=-1)
            input_cam_embed = input_cam_embed.expand(len(batch_latents), -1, -1)
            batch_cam_embed = torch.cat((batch_r, batch_t, batch_f), dim=-1)
            cam_embed = torch.cat((input_cam_embed, batch_cam_embed), dim=-1)
            clip_v_embed = torch.cat((clip_embed, cam_embed), dim=-1)
        
        else:
            #@ CONCAT V EMBED
            input_azimuth = batch['azimuth'][input_idx]
            input_elevation = -batch['elevation'][input_idx]
            batch_azimuth = batch['azimuth'][batch_idx]
            batch_elevation = -batch['elevation'][batch_idx]
            d_a = batch_azimuth - input_azimuth
            d_e = batch_elevation - input_elevation
            d_z = torch.zeros_like(d_a)
            v_embed = torch.stack([d_e, torch.sin(d_a), torch.cos(d_a), d_z], -1).unsqueeze(1)
            clip_v_embed = torch.cat([clip_embed, v_embed], -1)


        return batch_latents, batch_cameras, input_latents, input_cameras, clip_v_embed


    def embed_time(self, t):
        t_embed = timestep_embedding(t, self.time_embed_dim, repeat_only=False) # B,TED
        t_embed = self.time_embed(t_embed) # B,TED
        return t_embed


    def apply_model(self, noisy_latents, batch_cameras, input_latents, input_cameras, clip_v_embed, t, prev_depth=None, cfg_scale=1.0):
        '''
        Args:
            noisy_latents (B, 4, H, W)      - noisy latents
            batch_cameras (B, )             - noisy pytorch3d cameras
            input_latents (1, 4, H, W)      - input image latentes
            input_cameras (1, )             - input pytorch3d camera
            clip_embed (1, 1, 768)          - clip feature for input image
            t   (B, )                       - timestep

        Returns:
            pred (B, 4, H, W)               - predicted noise (or specified objective)
        '''

        #@ EMBED TIME
        batch_device = noisy_latents.device
        B = noisy_latents.shape[0]
        t_embed = self.embed_time(t)

        #@ GET VIEW ALIGNED FEATURES
        predict_mask = torch.ones((B,), device=batch_device)
        view_aligned_feat = self.view_attn(
                                noisy_latents=noisy_latents,
                                batch_cameras=batch_cameras,
                                predict_mask=predict_mask,
                                t_embed=t_embed,
                                t=t,
                                scheduler=self.scheduler,
                                overwrite_attn_depth=prev_depth,
                                input_latents=input_latents,
                                input_cameras=input_cameras
                            )
        #' noisy_latents (B, C, H, W)
        #' predict_mask (B, )
        #' t_embed (B, 256)
        #' view_aligned_feat (B, H, W, D, C)


        #@ RUN UNET
        input_latents = input_latents.expand(B, -1, -1, -1)
        clip_embed = self.cc_projection(clip_v_embed)
        if cfg_scale == 1.0:
            pred = self.unet_model(
                            noisy_latents,
                            t=t[:1],
                            clip_embed=clip_embed,
                            volume_feats=view_aligned_feat,
                            x_concat=input_latents,
                            is_train=True
                        )
        else:
            pred = self.unet_model.predict_with_unconditional_scale(
                            noisy_latents,
                            t=t[:1],
                            clip_embed=clip_embed,
                            volume_feats=view_aligned_feat,
                            x_concat=input_latents,
                            unconditional_scale=cfg_scale,
                        )
        #' pred (B, C, H, W)
        #' t (1, )
        #' clip_embed (B, 1, 768)

        return pred


    def sample(self, batch, trainer_config, cfg_scale, return_input=False, depth=False, verbose=True):

        batch_latents, batch_cameras, input_latents, input_cameras, clip_v_embed = self.prepare_batch(batch, trainer_config)

        if return_input:
            x_sample, intermediates = self.ddim.sample(batch_cameras, input_latents, input_cameras, clip_v_embed, unconditional_scale=cfg_scale, depth=depth, return_intermediates=True, verbose=verbose)
        else:
            x_sample = self.ddim.sample(batch_cameras, input_latents, input_cameras, clip_v_embed, unconditional_scale=cfg_scale, depth=depth, verbose=verbose)

        if return_input:
            return x_sample, batch_latents, input_latents, batch_cameras, intermediates
        return x_sample
    

    def p_losses(self, batch, trainer_config):

        #@ PREPARE BATCH
        batch_latents, batch_cameras, input_latents, input_cameras, clip_v_embed = self.prepare_batch(batch, trainer_config)

        #@ SAMPLE T AND BATCH
        B = batch_latents.shape[0]
        t = self.scheduler.sample_random_times(B, share_t=True, device=batch['images'].device)

        #@ ADD NOISE
        noisy_latents, noise = self.scheduler.q_sample(batch_latents.clone(), t=t)

        #@ APPLY MODEL
        if not self.feed_prev_depth:
            pred = self.apply_model(noisy_latents, batch_cameras, input_latents, input_cameras, clip_v_embed, t)
        else:
            prev_depth = input_latents[:,4:,...].clone().detach()
            pred = self.apply_model(noisy_latents, batch_cameras, input_latents, input_cameras, clip_v_embed, t, prev_depth=prev_depth)

        #@ SET LOSS TARGET
        if self.objective == 'noise':
            target = noise
        elif self.objective == 'x_start':
            target = batch_latents
        else:
            assert False, f'objective {self.objective} not implemented'

        #@ COMPUTE LOSS
        loss = self.loss_fn(target, pred).mean()

        return loss
    
    def forward(self, batch, trainer_config):
        
        #@ CALL P_LOSSES
        return self.p_losses(batch, trainer_config)

    def configure_optimizers(self, lr=None, verbose=False):

        if lr is None:
            lr = self.learning_rate
        
        if verbose:
            print(f'setting learning rate to {lr:.7f} ...')

        paras = []
        if self.finetune_projection:
            paras.append({"params": self.cc_projection.parameters(), "lr": lr},)

        paras.append({"params": self.unet_model.get_trainable_parameters(), "lr": lr},)
        paras.append({"params": self.time_embed.parameters(), "lr": lr*1.0},)
        paras.append({"params": self.view_attn.parameters(), "lr": lr*1.0},)

        opt = torch.optim.AdamW(paras, lr=lr)

        return opt