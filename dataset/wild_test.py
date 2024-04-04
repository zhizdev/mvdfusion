import os
import glob
import torch
import imageio
from skimage.transform import resize
import json
from einops import rearrange
import numpy as np
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
import cv2


def _unscale_depth( depths):
    shift = 0.5
    scale = 2.0
    depths = depths * scale + shift
    return depths

class GSO(torch.utils.data.Dataset):

    def __init__(self,
                 root='',
                 camera_type='fixed_set',
                 stage='train',
                 image_size=256,
                 sample_batch_size = None,
                 fix_elevation = True,
                 load_depth = False,
                 load_mask = False,
                 up_vec = 'y',
                 subset = 'test',
                 ):
        super().__init__()

        self.root = root
        self.camera_type = camera_type
        self.stage = stage
        self.image_size = image_size
        self.sample_batch_size = sample_batch_size
        self.fix_elevation = fix_elevation
        self.up_vec = up_vec
        subset_list_addr = f'{root}/{subset}.json'

        if os.path.exists(subset_list_addr):
            with open(subset_list_addr) as fp:
                self.subset_list = json.load(fp)
        else:
            self.subset_list = os.listdir(root)
        
        self.azimuths = torch.tensor([0.0, 0.39269909262657166, 0.7853981852531433, 1.1780972480773926, 1.5707963705062866, 
                1.9634953737258911, 2.356194496154785, 2.7488934993743896, 3.1415927410125732, 3.5342917442321777, 
                3.9269907474517822, 4.319689750671387, 4.71238899230957, 5.105088233947754, 5.497786998748779, 
                5.890486240386963])
        
        self.elevations = torch.tensor([0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 
                  0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 
                  0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 
                  0.5235987901687622, 0.5235987901687622])

        print(f'loaded {len(self.subset_list)} entries')
        
        self.__init_fixed_set_cameras()

    def __len__(self):
        return len(self.subset_list)
    
    def __getitem__(self, index):

        #@ LOAD SCENE FOLDER
        scene_dir = f'{self.root}/{self.subset_list[index]}'
        scene_list = glob.glob(scene_dir)

        assert len(scene_list) == 1

        #@ GET BATCH IDX
        batch_idx = torch.arange(0, 16)

        #@ LOAD DATA
        images = self._load_images(scene_list[0])
        images = images.expand(16, -1, -1, -1)

        R, T, f, c, azimuth, elevation = self._load_fixed_set_cameras(batch_idx)
        #@ RETURN DICT
        frame_dict = {
            'index': index,
            'idx': self.subset_list[index],
            'images':images,
            'R': R,
            'T': T,
            'f': f,
            'c': c,
            'azimuth': azimuth,
            'elevation': elevation,
        }

        return frame_dict

    def _load_images(self, img_path):

        rgb_list = []

        rgb_addr = img_path
        rgb = imageio.v3.imread(rgb_addr)
        rgb = resize(rgb, (self.image_size, self.image_size))
        rgb = torch.tensor(rgb, dtype=torch.float32)
        alpha = rgb[...,3:].clone()
        rgb = rgb[...,:3]
        rgb[...,0:1][alpha < 0.5] = 1.0
        rgb[...,1:2][alpha < 0.5] = 1.0
        rgb[...,2:3][alpha < 0.5] = 1.0
        rgb_list.append(rgb)

        images = torch.stack(rgb_list, dim=0)
        images = rearrange(images, 'b h w c -> b c h w')
        return images

    def __init_fixed_set_cameras(self):

        #@ INIT INTRINSICS
        distances = 1.5
        principal_point = ((0,0),)


        #@ INIT EXTRINSICS
        x = torch.cos(self.azimuths)*torch.cos(self.elevations)
        y = torch.sin(self.azimuths)*torch.cos(self.elevations)
        z = torch.sin(self.elevations)
        cam_pts = torch.stack([x,y,z],-1) * distances
        
        if self.up_vec == 'z':
            R, T = look_at_view_transform(eye=cam_pts, up=((0,0,1),))
        elif self.up_vec == '-z':
            R, T = look_at_view_transform(eye=cam_pts, up=((0,0,-1),))
        elif self.up_vec == 'y':
            R, T = look_at_view_transform(
                dist=distances,
                azim=self.azimuths * 180 / torch.pi + 90,
                elev=self.elevations * 180 / torch.pi,
                up=((0, 1, 0),),
            )
        else: raise NotImplementedError

        self.cameras_b32 = PerspectiveCameras(
                                R=R, 
                                T=T, 
                                focal_length = ((2.1875, 2.1875),),
                                principal_point=principal_point,
                            )   
    
    def _load_fixed_set_cameras(self, batch_idx):
        azimuth = self.azimuths[batch_idx]
        elevation = self.elevations[batch_idx]
        R = self.cameras_b32.R[batch_idx]
        T = self.cameras_b32.T[batch_idx]
        f = self.cameras_b32.focal_length[batch_idx]
        c = self.cameras_b32.principal_point[batch_idx]
        return R, T, f, c, azimuth, elevation
    