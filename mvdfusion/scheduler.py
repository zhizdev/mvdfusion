import torch
import torch.nn as nn



'''
Stable Diffusion comaptible DDPM scheduler
'''
class DDPMScheduler(nn.Module):

    def __init__(self, timesteps):
        super().__init__()

        self.num_timesteps = timesteps
        linear_start = 0.00085
        linear_end = 0.0120
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, self.num_timesteps, dtype=torch.float32) ** 2 # T
        assert betas.shape[0] == self.num_timesteps

        # all in float64 first
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # T
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # T
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_log_variance_clipped = torch.clamp(posterior_log_variance_clipped, min=-10)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped.float())

        self.register_buffer('_device', torch.tensor([0.]), persistent = False)
    
    def sample_random_times(self, b, share_t=True, device=None):

        if device is None:
            device = self._device.device

        time_steps = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        if share_t:
            time_steps = torch.zeros_like(time_steps) + time_steps[0]

        return time_steps
    
    def get_sampling_timesteps(self, num_steps=50):
        return

    def q_sample(self, x_start, t):
        B = x_start.shape[0]
        noise = torch.randn_like(x_start) # B,*

        sqrt_alphas_cumprod_  = self.sqrt_alphas_cumprod[t] # B,
        sqrt_one_minus_alphas_cumprod_ = self.sqrt_one_minus_alphas_cumprod[t] # B
        sqrt_alphas_cumprod_ = sqrt_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        sqrt_one_minus_alphas_cumprod_ = sqrt_one_minus_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        x_noisy = sqrt_alphas_cumprod_ * x_start + sqrt_one_minus_alphas_cumprod_ * noise
        return x_noisy, noise
    
    def predict_start_from_noise(self, x_noisy, eps, t):
        B = x_noisy.shape[0]

        sqrt_recip_alphas_cumprod_ = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1_alphas_cumprod_ = self.sqrt_recipm1_alphas_cumprod[t]
        sqrt_recip_alphas_cumprod_ = sqrt_recip_alphas_cumprod_.view(B, *[1 for _ in range(len(x_noisy.shape)-1)])
        sqrt_recipm1_alphas_cumprod_ = sqrt_recipm1_alphas_cumprod_.view(B, *[1 for _ in range(len(x_noisy.shape)-1)])
        x_start = sqrt_recip_alphas_cumprod_ * x_noisy - sqrt_recipm1_alphas_cumprod_ * eps
        return x_start
    