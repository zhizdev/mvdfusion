import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def disable_training_module(module: nn.Module):
    module = module.eval()
    module.train = disabled_train
    for para in module.parameters():
        para.requires_grad = False
    return module

def rank_zero_print(rank, text, **kwargs):
    if rank == 0:
        print(text, **kwargs)

def uncollate(batch, to_device=None):
    '''
    Iterate through dictionary to remove the first dimension iff 
    
    Args:
        batch (Dict) - batch[k] (1, B, C, H, W)

    Returns: 
        squeezed_batch (Dict) - squeezed_batch[k] (B, C, H, W)
    '''

    squeezed_batch = {}
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            assert batch[k].shape[0] == 1, f'uncollate does not support non singular shape {k}:{batch[k].shape}'
            squeezed_batch[k] = batch[k].squeeze(0).to(to_device) if to_device is not None else batch[k].squeeze(0)
        elif isinstance(batch[k], list):
            # print(f'uncollate does not support non singular shape {k}: {batch[k]}')
            # assert len(batch[k]) == 1, f'uncollate does not support non singular shape {k}: {batch[k]}'
            squeezed_batch[k] = batch[k][0]
        else:
            assert NotImplementedError, f'uncollate does not support {k}:{type(batch[k])}'
    return squeezed_batch

def dict_to_device(batch, to_device='cpu'):
    '''
    Move elements of dictionary to a device
    '''
    squeezed_batch = {}
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            squeezed_batch[k] = batch[k].to(to_device) if to_device is not None else batch[k].squeeze(0)
        elif isinstance(batch[k], list):
            squeezed_batch[k] = batch[k]
        else:
            assert NotImplementedError, f'uncollate does not support {k}:{type(batch[k])}'
    return squeezed_batch

def normalize(x):
    '''
    Normalize [0, 1] to [-1, 1]
    '''
    return torch.clip(x*2 - 1.0, -1.0, 1.0)

def unnormalize(x):
    '''
    Unnormalize [-1, 1] to [0, 1]
    '''
    return torch.clip((x + 1.0) / 2.0, 0.0, 1.0)

def split_list(a, n):
    '''
    Split list into n parts
    Args:
        a (list): list
        n (int): number of parts
    
    Returns:
        a_split (list[list]): nested list of a split into n parts
    '''
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

#@ From PyTorch3D
def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss

#@ From PyTorch3D
def sample_images_at_mc_locs(target_images, sampled_rays_ba, sampled_rays_xy):
    """
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this method samples the tensor `target_images` at the
    respective 2D locations.
    
    This function is used in order to extract the colors from
    ground truth images that correspond to the colors
    rendered using `MonteCarloRaysampler`.
    """
    #' target_images (B, C, H, W)

    ba = target_images.shape[0]
    dim = target_images.shape[1]
    spatial_size = sampled_rays_xy.shape[2:]
    assert target_images.shape[2] == target_images.shape[3]

    sampled_rays_xy = rearrange(sampled_rays_xy, 'n c -> () () n c').expand(ba, -1, -1, -1)

    images_sampled = torch.nn.functional.grid_sample(
        target_images, 
        -sampled_rays_xy,
        align_corners=True
    )
    images_sampled = rearrange(images_sampled[:,:,0,...], 'b c n -> b n c')

    aux_dim = torch.arange(len(sampled_rays_ba))
    gathered_target = images_sampled[sampled_rays_ba,aux_dim,...]

    return gathered_target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count


#@ FROM PyTorch3D
class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]:
            ```
            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]
            ```
        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.
        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`
        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`
        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.
        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (x, embed.sin(), embed.cos()
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
        self.append_input = append_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., (n_harmonic_functions * 2 + int(append_input)) * dim]
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        embed = torch.cat(
            (embed.sin(), embed.cos(), x)
            if self.append_input
            else (embed.sin(), embed.cos()),
            dim=-1,
        )
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int,
        n_harmonic_functions: int,
        append_input: bool,
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.
        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )