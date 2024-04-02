'''
Common camera utilities
'''

import math
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer.implicit.raysampling import _xy_to_ray_bundle


def _get_camera_slice(scene_cameras, indices):
        '''
        Return a subset of cameras from a super set given indices
        Args:
            scene_cameras (PyTorch3D Camera): cameras object
            indices (tensor or List): a flat list or tensor of indices
        Returns:
            camera_slice (PyTorch3D Camera) - cameras subset
        '''
        camera_slice = PerspectiveCameras(
            R = scene_cameras.R[indices], 
            T = scene_cameras.T[indices], 
            focal_length = scene_cameras.focal_length[indices],
            principal_point = scene_cameras.principal_point[indices],
            image_size = scene_cameras.image_size[indices] if scene_cameras.image_size is not None else None,
            device = scene_cameras.device,
        )
        return camera_slice

def _concat_cameras(camera_list):
        '''
        Returns a concatenation of a list of cameras
        Args:
            camera_list (List[PyTorch3D camera]): a list of PyTorch3D cameras
        '''
        R_list, T_list, f_list, c_list, size_list = [], [], [], [], []
        for cameras in camera_list:
            R_list.append(cameras.R)
            T_list.append(cameras.T)
            f_list.append(cameras.focal_length)
            c_list.append(cameras.principal_point)
            if cameras.image_size is not None:
                size_list.append(cameras.image_size)

        camera_slice = PerspectiveCameras(
            R = torch.cat(R_list), 
            T = torch.cat(T_list), 
            focal_length = torch.cat(f_list),
            principal_point = torch.cat(c_list),
            # image_size = torch.cat(size_list) if len(size_list) > 0 else None,
            device = camera_list[0].device,
        )
        return camera_slice

def _get_relative_camera(scene_cameras:PerspectiveCameras, query_idx=None, query_camera=None, center_at_origin=False, shift_z=False):
        """
        Transform context cameras relative to a base query camera
        Args:
            scene_cameras (PyTorch3D Camera): cameras object
            query_idx (tensor or List): a length 1 list defining query idx
        Returns:
            cams_relative (PyTorch3D Camera): cameras object relative to query camera
        """
        if center_at_origin is False:
            assert shift_z is False

        if query_idx is not None:
            query_camera = _get_camera_slice(scene_cameras, query_idx)
        else:
            assert query_camera is not None

        if shift_z:
            optical_center = _get_nearest_centroid(scene_cameras).to(scene_cameras.device)
            z_dist = torch.linalg.norm(query_camera.get_camera_center()[0] - optical_center.unsqueeze(0), axis=1)

        
        query_world2view = query_camera.get_world_to_view_transform()
        all_world2view = scene_cameras.get_world_to_view_transform()
        
        if center_at_origin:
            identity_cam = PerspectiveCameras(device=scene_cameras.device, R=query_camera.R, T=query_camera.T)
        else:
            T = torch.zeros((1, 3))
            identity_cam = PerspectiveCameras(device=scene_cameras.device, R=query_camera.R, T=T)
         
        identity_world2view  = identity_cam.get_world_to_view_transform()

        # compose the relative transformation as g_i^{-1} g_j
        relative_world2view = identity_world2view.inverse().compose(all_world2view)
        
        # generate a camera from the relative transform
        relative_matrix = relative_world2view.get_matrix()
        cams_relative = PerspectiveCameras(
                            R = relative_matrix[:, :3, :3],
                            T = relative_matrix[:, 3, :3],
                            focal_length = scene_cameras.focal_length,
                            principal_point = scene_cameras.principal_point,
                            image_size = scene_cameras.image_size,
                            device = scene_cameras.device,
                        )
        
        if shift_z:
            scene_shift = (cams_relative.R.permute(0,2,1)@(torch.tensor([[0.0],[0.0],[z_dist]], device=scene_cameras.device)))[:,:,0]
            cams_relative = PerspectiveCameras(
                R = cams_relative.R, 
                T = cams_relative.T + scene_shift, 
                focal_length = cams_relative.focal_length,
                principal_point = cams_relative.principal_point,
                image_size = cams_relative.image_size,
                device = cams_relative.device,
            )
        return cams_relative

def _normalize_camera(scene_cameras: PerspectiveCameras, distance=3.5, look_at_optical_center=False):
        '''
        Normalize the camera origins to be roughly distance 1.5 away from origin
        '''
        if look_at_optical_center:
            look_at_center = _get_nearest_centroid(scene_cameras).unsqueeze(-1).unsqueeze(0)
            scene_shift = (scene_cameras.R.permute(0,2,1)@(look_at_center))[:,:,0]
            scene_cameras = PerspectiveCameras(
                R = scene_cameras.R, 
                T = scene_cameras.T - scene_shift, 
                focal_length = scene_cameras.focal_length,
                principal_point = scene_cameras.principal_point,
                image_size = scene_cameras.image_size,
                device = scene_cameras.device,
            )
             
        cam_dist_mean = torch.mean(torch.linalg.norm(scene_cameras.get_camera_center(), axis=1))
        scale_factor = 1.0 / (cam_dist_mean / distance)

        normalized_cameras = PerspectiveCameras(
            R = scene_cameras.R, 
            T = scene_cameras.T * scale_factor, 
            focal_length = scene_cameras.focal_length,
            principal_point = scene_cameras.principal_point,
            image_size = scene_cameras.image_size,
            device = scene_cameras.device,
        )
        return normalized_cameras

#@ https://www.crewes.org/Documents/ResearchReports/2010/CRR201032.pdf
def _get_nearest_centroid(cameras: PerspectiveCameras):
    '''
    Given PyTorch3D cameras, find the nearest point along their principal ray
    '''

    #@ GET CAMERA CENTERS AND DIRECTIONS
    camera_centers = cameras.get_camera_center()

    c_mean = (cameras.principal_point).mean(dim=0)
    xy_grid = c_mean.unsqueeze(0).unsqueeze(0)
    ray_vis = _xy_to_ray_bundle(cameras, xy_grid.expand(len(cameras),-1,-1), 1.0, 15.0, 20, True)
    camera_directions = ray_vis.directions

    #@ CONSTRUCT MATRICIES
    A = torch.zeros((3*len(cameras)), len(cameras)+3)
    b = torch.zeros((3*len(cameras), 1))
    A[:,:3] = torch.eye(3).repeat(len(cameras),1)
    for ci in range(len(camera_directions)):
        A[3*ci:3*ci+3, ci+3] = -camera_directions[ci]
        b[3*ci:3*ci+3, 0] = camera_centers[ci]
    #' A (3*N, 3*N+3)   b (3*N, 1)

    #@ SVD
    U, s, VT = torch.linalg.svd(A)
    Sinv = torch.diag(1/s)
    if len(s) < 3*len(cameras):
        Sinv = torch.cat((Sinv, torch.zeros((Sinv.shape[0], 3*len(cameras) - Sinv.shape[1]), device=Sinv.device)), dim=1)
    x = torch.matmul(VT.T, torch.matmul(Sinv,torch.matmul(U.T, b)))
    
    centroid = x[:3,0]
    return centroid

def _get_relative_v(cameras: PerspectiveCameras, input_idx):

    return