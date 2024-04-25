
import os
import imageio
import torch
import matplotlib.pyplot as plt

from pytorch3d.renderer import (PerspectiveCameras, look_at_view_transform, FoVOrthographicCameras,)
import numpy as np
import math
from einops import rearrange
from utils.common_utils import normalize, unnormalize
from utils.camera_utils import RelativeCameraLoader, _get_relative_camera, _normalize_camera, _get_camera_slice

def get_cam_loc(theta, phi, cam_dist):
    x = cam_dist * math.sin(phi) * math.cos(theta)
    y = cam_dist * math.sin(phi) * math.sin(theta)
    z = cam_dist * math.cos(phi)
    return ((x, z, -y), )

@torch.no_grad()
def visualize_image_depth_diffusion(config, model, batch, global_step, loss_list=None, concat_input=False, overwrite_x_noisy=True):

    g_cpu = torch.Generator()
    g_cpu.manual_seed(global_step)

    #@ PREPARE BATCH
    batch_latents, batch_cameras, input_latents, input_cameras, clip_embed = model.module.prepare_batch(batch, dict(config['trainer']), generator=g_cpu)
    
    if not config['saver'].get('regression', False):
        pred_x0 = model.module.ddim.sample(batch_cameras, input_latents, input_cameras, clip_embed, unconditional_scale=1.0, depth=True)
    else:
        t = model.module.scheduler.sample_random_times(batch_latents.shape[0], share_t=True, device=batch['images'].device)
        t = torch.ones_like(t) * 999
        pred_noise = model.module.apply_model(batch_latents, batch_cameras, input_latents, input_cameras, clip_embed, t=t)
        pred_x0 = model.module.scheduler.predict_start_from_noise(batch_latents, pred_noise, t=t)
        pred_x0 = pred_x0[:,:4,...]

    pred_rgb = model.module.decode(pred_x0[:,:4,...])
    
    #@ SAVE JPG
    save_dir = config['saver']['exp_dir'] + config['saver']['vis_dir']
    os.makedirs(save_dir, exist_ok=True)

    jpg_dir = f'{save_dir}/{global_step:07d}.jpg'

    input_rgb = model.module.decode(input_latents[:,:4,...])
    vis_input = rearrange(input_rgb, 'b c h w -> b h w c').detach().cpu().numpy()
    vis_input = np.hstack(vis_input)

    noisy_latent = torch.randn_like(pred_x0)
    if overwrite_x_noisy:
        noisy_latent[0] = batch_latents[0]
    noisy_rgb = model.module.decode(noisy_latent[:,:4,...])
    vis_noise = rearrange(noisy_rgb, 'b c h w -> b h w c').detach().cpu().numpy()
    vis_noise = np.hstack(vis_noise)
    if concat_input:
        vis_noise = np.hstack((vis_input, vis_noise))
    
    vis_pred = rearrange(pred_rgb, 'b c h w -> b h w c').detach().cpu().numpy()
    vis_pred = np.hstack(vis_pred)
    if concat_input:
        vis_pred = np.hstack((vis_input, vis_pred))

    gt_rgb = model.module.decode(batch_latents[:,:4,...])
    vis_gt = rearrange(gt_rgb, 'b c h w -> b h w c').detach().cpu().numpy()
    vis_gt = np.hstack(vis_gt)
    if concat_input:
        vis_gt = np.hstack((vis_input, vis_gt))


    #@ DEPTHS
    vis_input_depth = torch.nn.functional.interpolate(unnormalize(input_latents[:,4:,...]), scale_factor=8.0)
    vis_input_depth = rearrange(vis_input_depth, 'b c h w -> b h w c').expand(-1,-1,-1,3).detach().cpu().numpy()
    vis_input_depth = np.hstack(vis_input_depth)

    gt_depth = torch.nn.functional.interpolate(unnormalize(batch_latents[:,4:,...]), scale_factor=8.0)
    vis_gt_depth = rearrange(gt_depth, 'b c h w -> b h w c').expand(-1,-1,-1,3).detach().cpu().numpy()
    vis_gt_depth = np.hstack(vis_gt_depth)
    if concat_input:
        vis_gt_depth = np.hstack((vis_input_depth, vis_gt_depth))
    vis_gt_depth = np.clip(vis_gt_depth, 0.0, 1.0)

    pred_depth = torch.nn.functional.interpolate(unnormalize(pred_x0[:,4:,...]), scale_factor=8.0)
    vis_depth = rearrange(pred_depth, 'b c h w -> b h w c').expand(-1,-1,-1,3).detach().cpu().numpy()
    vis_depth = np.hstack(vis_depth)
    if concat_input:
        vis_depth = np.hstack((vis_input_depth, vis_depth))
    vis_depth = np.clip(vis_depth, 0.0, 1.0)
    

    print(vis_gt.shape, vis_depth.shape, vis_gt_depth.shape)
    vis = np.vstack((vis_noise, vis_pred, vis_gt, vis_depth, vis_gt_depth))
    imageio.imwrite(jpg_dir, (vis*255).astype(np.uint8))

    #@ PLOT LOSS
    if loss_list is not None:
        loss_dir = config['saver']['exp_dir'] + '/' + config['saver']['loss_dir']
        os.makedirs(loss_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label='loss')
        plt.plot(loss_list[-100:], label='loss_recent')
        plt.legend()
        plt.savefig(f'{loss_dir}/_loss.png')
        plt.close()