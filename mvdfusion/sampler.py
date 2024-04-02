import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from external.sd1.ldm.modules.diffusionmodules.util import make_ddim_timesteps, timestep_embedding

def repeat_to_batch(tensor, B, VN):
    t_shape = tensor.shape
    ones = [1 for _ in range(len(t_shape)-1)]
    tensor_new = tensor.view(B,1,*t_shape[1:]).repeat(1,VN,*ones).view(B*VN,*t_shape[1:])
    return tensor_new

class DDIMSampler:
    def __init__(self, model, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., latent_size=32, overwrite_x_noisy=False, z_dim=4, feed_prev_depth=False):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.scheduler.num_timesteps
        self.latent_size = latent_size
        self._make_schedule(ddim_num_steps, ddim_discretize, ddim_eta)
        self.eta = ddim_eta
        self.overwrite_x_noisy = overwrite_x_noisy
        self.z_dim = z_dim
        self.feed_prev_depth = feed_prev_depth

    def _make_schedule(self,  ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose) # DT
        ddim_timesteps_ = torch.from_numpy(self.ddim_timesteps.astype(np.int64)) # DT

        alphas_cumprod = self.model.scheduler.alphas_cumprod # T
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        self.ddim_alphas = alphas_cumprod[ddim_timesteps_].double() # DT
        self.ddim_alphas_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[ddim_timesteps_[:-1]]], 0) # DT
        self.ddim_sigmas = ddim_eta * torch.sqrt((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev))

        self.ddim_alphas_raw = self.model.scheduler.alphas[ddim_timesteps_].float() # DT
        self.ddim_sigmas = self.ddim_sigmas.float()
        self.ddim_alphas = self.ddim_alphas.float()
        self.ddim_alphas_prev = self.ddim_alphas_prev.float()
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas).float()


    @torch.no_grad()
    def denoise_apply_impl(self, x_target_noisy, index, noise_pred, is_step0=False):
        """
        @param x_target_noisy: B,N,4,H,W
        @param index:          index
        @param noise_pred:     B,N,4,H,W
        @param is_step0:       bool
        @return:
        """
        device = x_target_noisy.device
        #B,N,_,H,W = x_target_noisy.shape

        # apply noise
        a_t = self.ddim_alphas[index].to(device).float().view(1,1,1,1)
        a_prev = self.ddim_alphas_prev[index].to(device).float().view(1,1,1,1)
        sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index].to(device).float().view(1,1,1,1)
        sigma_t = self.ddim_sigmas[index].to(device).float().view(1,1,1,1)

        pred_x0 = (x_target_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
        dir_xt = torch.clamp(1. - a_prev - sigma_t**2, min=1e-7).sqrt() * noise_pred
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        if not is_step0:
            noise = sigma_t * torch.randn_like(x_target_noisy)
            x_prev = x_prev + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def denoise_apply(self, x_target_noisy, batch_cameras, input_latents, input_cameras, clip_embed, time_steps, index, is_step0=False, prev_depth=None, cfg_scale=1.0):
        """
        @param x_target_noisy:   B,N,4,H,W
        @param input_info:
        @param clip_embed:       B,M,768
        @param time_steps:       B,
        @param index:            int
        @param unconditional_scale:
        @param batch_view_num:   int
        @param is_step0:         bool
        @return:
        """
        B, C, H, W = x_target_noisy.shape

        if self.feed_prev_depth:
            noise = self.model.apply_model(x_target_noisy, batch_cameras, input_latents, input_cameras, clip_embed, time_steps, prev_depth=prev_depth, cfg_scale=cfg_scale)
        else:
            noise = self.model.apply_model(x_target_noisy, batch_cameras, input_latents, input_cameras, clip_embed, time_steps, cfg_scale=cfg_scale)            
        x_prev, pred_x0 = self.denoise_apply_impl(x_target_noisy, index, noise, is_step0)
        return x_prev, pred_x0

    @torch.no_grad()
    def sample(self, batch_cameras, input_latents, input_cameras, clip_embed, unconditional_scale=1.0, depth=False, return_intermediates=False, verbose=True):
        """
        @param input_info:      x, elevation
        @param clip_embed:      B,M,768
        @param unconditional_scale:
        @param log_every_t:
        @param batch_view_num:
        @return:
        """
        print(f"unconditional scale {unconditional_scale:.1f}")
        C, H, W = self.z_dim, self.latent_size, self.latent_size
        B = clip_embed.shape[0]
        device = self.model._device.device
        if not depth:
            x_target_noisy = torch.randn([B, C, H, W], device=device)
        else:
            x_target_noisy = torch.randn([B, C+1, H, W], device=device)

        if self.overwrite_x_noisy:
            x_target_noisy[0] = input_latents[0].clone()

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        prev_depth = None
        intermediates = []
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)if verbose else time_range
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # index in ddim state
            time_steps = torch.full((B,), step, device=device, dtype=torch.long)

            if self.overwrite_x_noisy:
                x_target_noisy[0] = input_latents[0].clone()

            x_target_noisy, pred_x0 = self.denoise_apply(
                                            x_target_noisy, 
                                            batch_cameras, 
                                            input_latents, 
                                            input_cameras, 
                                            clip_embed, 
                                            time_steps, 
                                            index, 
                                            is_step0=index==0,
                                            prev_depth=prev_depth if self.feed_prev_depth else None,
                                            cfg_scale=unconditional_scale
                                            )
            
            #@ FETCH DEPTH ESTIMATE
            prev_depth = pred_x0[:,4:,...].clone().detach()
            
            intermediates.append({'t':step, 'xt':x_target_noisy, 'x0':pred_x0})
        x_output = x_target_noisy

        if return_intermediates:
            return x_output, intermediates

        return x_output