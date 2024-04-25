#@ Modified from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/implicit/utils.py

import warnings
from typing import Optional, Tuple, Union, NamedTuple
import dataclasses

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.ops import padded_to_packed
from pytorch3d.renderer.cameras import CamerasBase
from torch.nn import functional as F
from einops import rearrange

class RayBundle(NamedTuple):
    """
    Parametrizes points along projection rays by storing:

        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
    """

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor


@dataclasses.dataclass
class HeterogeneousRayBundle:
    """
    Members:
        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
        camera_ids: A tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of unique sampled cameras.
        camera_counts: A tensor of shape (N, ) which how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts)==total_number_of_rays`

    If we sample cameras of ids [0, 3, 5, 3, 1, 0, 0] that would be
    stored as camera_ids=[1, 3, 5, 0] and camera_counts=[1, 2, 1, 3]. `camera_ids` is a
    set like object with no particular ordering of elements. ith element of
    `camera_ids` coresponds to the ith element of `camera_counts`.
    """

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor
    camera_ids: Optional[torch.LongTensor] = None
    camera_counts: Optional[torch.LongTensor] = None


def ray_bundle_to_ray_points(
    ray_bundle: Union[RayBundle, HeterogeneousRayBundle]
) -> torch.Tensor:
    """
    Converts rays parametrized with a `ray_bundle` (an instance of the `RayBundle`
    named tuple or HeterogeneousRayBundle dataclass) to 3D points by
    extending each ray according to the corresponding length.

    E.g. for 2 dimensional tensors `ray_bundle.origins`, `ray_bundle.directions`
        and `ray_bundle.lengths`, the ray point at position `[i, j]` is::

            ray_bundle.points[i, j, :] = (
                ray_bundle.origins[i, :]
                + ray_bundle.directions[i, :] * ray_bundle.lengths[i, j]
            )

    Note that both the directions and magnitudes of the vectors in
    `ray_bundle.directions` matter.

    Args:
        ray_bundle: A `RayBundle` or `HeterogeneousRayBundle` object with fields:
            origins: A tensor of shape `(..., 3)`
            directions: A tensor of shape `(..., 3)`
            lengths: A tensor of shape `(..., num_points_per_ray)`

    Returns:
        rays_points: A tensor of shape `(..., num_points_per_ray, 3)`
            containing the points sampled along each ray.
    """
    return ray_bundle_variables_to_ray_points(
        ray_bundle.origins, ray_bundle.directions, ray_bundle.lengths
    )

def _jiggle_within_stratas(bin_centers: torch.Tensor) -> torch.Tensor:
    """
    Performs sampling of 1 point per bin given the bin centers.

    More specifically, it replaces each point's value `z`
    with a sample from a uniform random distribution on
    `[z - delta_-, z + delta_+]`, where `delta_-` is half of the difference
    between `z` and the previous point, and `delta_+` is half of the difference
    between the next point and `z`. For the first and last items, the
    corresponding boundary deltas are assumed zero.

    Args:
        `bin_centers`: The input points of size (..., N); the result is broadcast
            along all but the last dimension (the rows). Each row should be
            sorted in ascending order.

    Returns:
        a tensor of size (..., N) with the locations jiggled within stratas/bins.
    """
    # Get intervals between bin centers.
    mids = 0.5 * (bin_centers[..., 1:] + bin_centers[..., :-1])
    upper = torch.cat((mids, bin_centers[..., -1:]), dim=-1)
    lower = torch.cat((bin_centers[..., :1], mids), dim=-1)
    # Samples in those intervals.
    jiggled = lower + (upper - lower) * torch.rand_like(lower)
    return jiggled

def _custom_xy_to_ray_bundle(
    cameras: CamerasBase,
    xy_grid: torch.Tensor,
    rays_zs: torch.Tensor,
    n_pts_per_ray: int,
    unit_directions: bool,
) -> RayBundle:
    """
    Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
    This adds to each xy location in the grid a vector of `n_pts_per_ray` depths
    uniformly spaced between `min_depth` and `max_depth`.

    The extended grid is then unprojected with `cameras` to yield
    ray origins, directions and depths.

    Args:
        cameras: cameras object representing a batch of cameras.
        xy_grid: torch.tensor grid of image xy coords.
        min_depth: The minimum depth of each ray-point.
        max_depth: The maximum depth of each ray-point.
        n_pts_per_ray: The number of points sampled along each ray.
        unit_directions: whether to normalize direction vectors in ray bundle.
        stratified_sampling: if True, performs stratified sampling in n_pts_per_ray
            bins for each ray; otherwise takes n_pts_per_ray deterministic points
            on each ray with uniform offsets.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()

    # ray z-coords
    rays_zs = rays_zs
    # rays_zs = xy_grid.new_empty((0,))
    # if n_pts_per_ray > 0:
    #     depths = torch.linspace(
    #         min_depth,
    #         max_depth,
    #         n_pts_per_ray,
    #         dtype=xy_grid.dtype,
    #         device=xy_grid.device,
    #     )
    #     rays_zs = depths[None, None].expand(batch_size, n_rays_per_image, n_pts_per_ray)

    #     if stratified_sampling:
    #         rays_zs = _jiggle_within_stratas(rays_zs)

    # make two sets of points at a constant depth=1 and 2
    to_unproject = torch.cat(
        (
            xy_grid.view(batch_size, 1, n_rays_per_image, 2)
            .expand(batch_size, 2, n_rays_per_image, 2)
            .reshape(batch_size, n_rays_per_image * 2, 2),
            torch.cat(
                (
                    xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                    2.0 * xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                ),
                dim=1,
            ),
        ),
        dim=-1,
    )

    # unproject the points
    unprojected = cameras.unproject_points(to_unproject, from_ndc=True)

    # split the two planes back
    rays_plane_1_world = unprojected[:, :n_rays_per_image]
    rays_plane_2_world = unprojected[:, n_rays_per_image:]

    # directions are the differences between the two planes of points
    rays_directions_world = rays_plane_2_world - rays_plane_1_world

    # origins are given by subtracting the ray directions from the first plane
    rays_origins_world = rays_plane_1_world - rays_directions_world

    if unit_directions:
        rays_directions_world = F.normalize(rays_directions_world, dim=-1)

    return RayBundle(
        rays_origins_world.view(batch_size, *spatial_size, 3),
        rays_directions_world.view(batch_size, *spatial_size, 3),
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,
    )


class DepthBasedMultinomialRaysampler(torch.nn.Module):

    def __init__(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        n_rays_per_image: Optional[int] = None,
        n_rays_total: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
    ) -> None:
        """
        Args:
            min_x: The leftmost x-coordinate of each ray's source pixel's center.
            max_x: The rightmost x-coordinate of each ray's source pixel's center.
            min_y: The topmost y-coordinate of each ray's source pixel's center.
            max_y: The bottommost y-coordinate of each ray's source pixel's center.
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
            n_rays_per_image: If given, this amount of rays are sampled from the grid.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            n_rays_total: How many rays in total to sample from the cameras provided. The result
                is as if `n_rays_total_training` cameras were sampled with replacement from the
                cameras provided and for every camera one ray was sampled. If set returns the
                HeterogeneousRayBundle with batch_size=n_rays_total.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            unit_directions: whether to normalize direction vectors in ray bundle.
            stratified_sampling: if True, performs stratified random sampling
                along the ray; otherwise takes ray points at deterministic offsets.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._n_rays_per_image = n_rays_per_image
        self._n_rays_total = n_rays_total
        self._unit_directions = unit_directions
        self._stratified_sampling = stratified_sampling
        self.min_x, self.max_x = min_x, max_x
        self.min_y, self.max_y = min_y, max_y
        # get the initial grid of image xy coords
        y, x = meshgrid_ij(
            torch.linspace(min_y, max_y, image_height, dtype=torch.float32),
            torch.linspace(min_x, max_x, image_width, dtype=torch.float32),
        )
        _xy_grid = torch.stack([x, y], dim=-1)

        self.register_buffer("_xy_grid", _xy_grid, persistent=False)


    def forward(
        self,
        cameras: CamerasBase,
        depth_channel: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        n_rays_per_image: Optional[int] = None,
        n_pts_per_ray: Optional[int] = None,
        stratified_sampling: Optional[bool] = None,
        n_rays_total: Optional[int] = None,
        **kwargs,
    ) -> Union[RayBundle, HeterogeneousRayBundle]:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            mask: if given, the rays are sampled from the mask. Should be of size
                (batch_size, image_height, image_width).
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
            n_rays_per_image: If given, this amount of rays are sampled from the grid.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            n_pts_per_ray: The number of points sampled along each ray.
            stratified_sampling: if set, overrides stratified_sampling provided
                in __init__.
            n_rays_total: How many rays in total to sample from the cameras provided. The result
                is as if `n_rays_total_training` cameras were sampled with replacement from the
                cameras provided and for every camera one ray was sampled. If set returns the
                HeterogeneousRayBundle with batch_size=n_rays_total.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
        Returns:
            A named tuple RayBundle or dataclass HeterogeneousRayBundle with the
            following fields:

            origins: A tensor of shape
                `(batch_size, s1, s2, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, s1, s2, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, s1, s2, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, s1, s2, 2)`
                containing the 2D image coordinates of each ray or,
                if mask is given, `(batch_size, n, 1, 2)`
            Here `s1, s2` refer to spatial dimensions.
            `(s1, s2)` refer to (highest priority first):
                - `(1, 1)` if `n_rays_total` is provided, (batch_size=n_rays_total)
                - `(n_rays_per_image, 1) if `n_rays_per_image` if provided,
                - `(n, 1)` where n is the minimum cardinality of the mask
                        in the batch if `mask` is provided
                - `(image_height, image_width)` if nothing from above is satisfied

            `HeterogeneousRayBundle` has additional members:
                - camera_ids: tensor of shape (M,), where `M` is the number of unique sampled
                    cameras. It represents unique ids of sampled cameras.
                - camera_counts: tensor of shape (M,), where `M` is the number of unique sampled
                    cameras. Represents how many times each camera from `camera_ids` was sampled

            `HeterogeneousRayBundle` is returned if `n_rays_total` is provided else `RayBundle`
            is returned.
        """
        n_rays_total = n_rays_total or self._n_rays_total
        n_rays_per_image = n_rays_per_image or self._n_rays_per_image
        if (n_rays_total is not None) and (n_rays_per_image is not None):
            raise ValueError(
                "`n_rays_total` and `n_rays_per_image` cannot both be defined."
            )

        camera_ids: torch.LongTensor = torch.arange(len(cameras), dtype=torch.long)

        batch_size = cameras.R.shape[0]
        device = cameras.device

        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device).expand(batch_size, -1, -1, -1)

        if mask is not None and n_rays_per_image is None:
            # if num rays not given, sample according to the smallest mask
            n_rays_per_image = (
                n_rays_per_image or mask.sum(dim=(1, 2)).min().int().item()
            )

        n_pts_per_ray = (
            n_pts_per_ray if n_pts_per_ray is not None else self._n_pts_per_ray
        )
        stratified_sampling = (
            stratified_sampling
            if stratified_sampling is not None
            else self._stratified_sampling
        )

        #@ CUSTOM RAYZS
        #' (B, n_rays_per_image, n_pts_per_ray)
        n_pts_per_ray = depth_channel.shape[1]
        rays_zs = depth_channel
        rays_zs = rearrange(depth_channel, 'b n h w -> b (h w) n')

        ray_bundle = _custom_xy_to_ray_bundle(
            cameras,
            xy_grid,
            rays_zs=rays_zs,
            n_pts_per_ray=n_pts_per_ray,
            unit_directions=self._unit_directions,
        )

        return  ray_bundle
        

