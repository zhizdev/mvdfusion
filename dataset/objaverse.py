import os
import glob
import torch
import imageio
import json
from einops import rearrange
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform


AZIMUTHS_B64 = [0.39269909262657166, 1.1780972480773926, 1.9634954929351807, 2.7488934993743896, 
                3.5342917442321777, 4.319689750671387, 5.105088233947754, 5.890486240386963, 0.0, 
                0.39269909262657166, 0.7853981852531433, 1.1780972480773926, 1.5707963705062866, 
                1.9634953737258911, 2.356194496154785, 2.7488934993743896, 3.1415927410125732, 3.5342917442321777, 
                3.9269907474517822, 4.319689750671387, 4.71238899230957, 5.105088233947754, 5.497786998748779, 
                5.890486240386963, 0.39269909262657166, 1.1780972480773926, 1.9634954929351807, 2.7488934993743896, 
                3.5342917442321777, 4.319689750671387, 5.105088233947754, 5.890486240386963, 0.0, 0.7853981852531433, 
                1.5707963705062866, 2.356194496154785, 3.1415927410125732, 3.9269907474517822, 4.71238899230957, 
                5.497786998748779, 0.0, 0.39269909262657166, 0.7853981852531433, 1.1780972480773926, 1.5707963705062866, 
                1.9634953737258911, 2.356194496154785, 2.7488934993743896, 3.1415927410125732, 3.5342917442321777, 
                3.9269907474517822, 4.319689750671387, 4.71238899230957, 5.105088233947754, 5.497786998748779, 
                5.890486240386963, 0.39269909262657166, 1.1780972480773926, 1.9634954929351807, 2.7488934993743896, 
                3.5342917442321777, 4.319689750671387, 5.105088233947754, 5.890486240386963]

ELEVATIONS_B64 = [-0.1745329201221466, -0.1745329201221466, -0.1745329201221466, -0.1745329201221466, 
                  -0.1745329201221466, -0.1745329201221466, -0.1745329201221466, -0.1745329201221466, 
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                  0.1745329201221466, 0.1745329201221466, 0.1745329201221466, 0.1745329201221466, 0.1745329201221466, 
                  0.1745329201221466, 0.1745329201221466, 0.1745329201221466, 0.3490658402442932, 0.3490658402442932, 
                  0.3490658402442932, 0.3490658402442932, 0.3490658402442932, 0.3490658402442932, 0.3490658402442932, 
                  0.3490658402442932, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 
                  0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 
                  0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 
                  0.5235987901687622, 0.5235987901687622, 0.6981316804885864, 0.6981316804885864, 0.6981316804885864, 
                  0.6981316804885864, 0.6981316804885864, 0.6981316804885864, 0.6981316804885864, 0.6981316804885864]

def _unscale_depth( depths):
    shift = 0.5
    scale = 2.0
    depths = depths * scale + shift
    return depths

class Objaverse(torch.utils.data.Dataset):

    def __init__(self,
                 root='',
                 subset='400k',
                 camera_type='fixed_set',
                 stage='train',
                 image_size=256,
                 sample_batch_size = None,
                 fix_elevation = True,
                 load_depth = False,
                 load_mask = False,
                 up_vec = 'y',
                 ):
        super().__init__()

        self.root = root
        self.subset  = subset
        self.camera_type = camera_type
        self.stage = stage
        self.image_size = image_size
        self.sample_batch_size = sample_batch_size
        self.fix_elevation = fix_elevation
        self.load_depth = load_depth
        self.load_mask = load_mask
        self.up_vec = up_vec

        subset_list_addr = f'{root}/subset_list/{subset}_{stage}.json'
        assert os.path.exists(subset_list_addr), 'subset not found'

        with open(subset_list_addr) as fp:
            self.subset_list = json.load(fp)

        print(f'loaded {len(self.subset_list)} entries from {subset}|{stage}')

        self.azimuths_b64 = torch.tensor(AZIMUTHS_B64)
        self.elevations_b64 = torch.tensor(ELEVATIONS_B64)
        self.__init_fixed_set_cameras()

    def __len__(self):
        return len(self.subset_list)
    
    def __getitem__(self, index):

        #@ LOAD SCENE FOLDER
        scene_dir = f'{self.root}/{self.subset}/{self.subset_list[index]}/views/'
        scene_list = glob.glob(scene_dir + '*_rgb.jpg')

        if self.camera_type == 'fixed_set':
            assert len(scene_list) == 64

        #@ GET BATCH IDX
        if self.fix_elevation:
            # batch_idx = torch.arange(8,8+16, step=2)
            if self.stage == 'train':
                batch_idx = torch.arange(8+16+8+8, 8+16+8+8+16, step=1) #step 2
            else:
                batch_idx = torch.arange(8+16+8+8, 8+16+8+8+16, step=1)

        else:
            if self.stage == 'test' or self.sample_batch_size == None:
                batch_idx = torch.arange(len(scene_list))
            elif self.sample_batch_size is not None:
                batch_idx = torch.randperm(scene_list)[:self.sample_batch_size]
            else:
                raise NotImplementedError
                

        #@ LOAD DATA
        images, masks, depths = self._load_images(scene_dir, batch_idx)

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

        if self.load_depth:
            frame_dict['depths'] = depths

        if self.load_mask:
            frame_dict['masks'] = masks

        return frame_dict

    def _load_images(self, image_dir, batch_idx):

        rgb_list = []
        mask_list = []
        depth_list = []

        for idx in batch_idx:
            rgb_addr = f'{image_dir}/{idx:03d}_rgb.jpg'
            rgb = torch.tensor(imageio.v3.imread(rgb_addr), dtype=torch.float32) / 255.0
            rgb_list.append(rgb)

            if self.load_depth or self.load_mask:
                depth_addr = f'{image_dir}/{idx:03d}_depth.png'
                depth = torch.tensor(imageio.v3.imread(depth_addr), dtype=torch.float32) / 255.0
                depth_list.append(depth)

            if self.load_mask:
                mask_addr = f'{image_dir}/{idx:03d}_mask.jpg'
                mask = torch.tensor(imageio.v3.imread(mask_addr), dtype=torch.float32) / 255.0
                mask_list.append(mask)

        images = torch.stack(rgb_list, dim=0)
        images = rearrange(images, 'b h w c -> b c h w')

        depths = None
        if self.load_depth:
            depths = torch.stack(depth_list, dim=0)
            depths = rearrange(depths[...,:1], 'b h w c -> b c h w')

        masks = None
        if self.load_mask:
            masks = torch.stack(mask_list, dim=0)
            masks = rearrange(masks, 'b h w -> b () h w')
        
        return images, masks, depths

    def _load_fixed_set_cameras(self, batch_idx):

        azimuth = self.azimuths_b64[batch_idx]
        elevation = self.elevations_b64[batch_idx]
        
        R = self.cameras_b64.R[batch_idx]
        T = self.cameras_b64.T[batch_idx]
        f = self.cameras_b64.focal_length[batch_idx]
        c = self.cameras_b64.principal_point[batch_idx]

        return R, T, f, c, azimuth, elevation
    
    def _normalize_depths(self, depths):
        
        shift = 0.5
        scale = 2.0
        depths = depths * scale + shift
        return depths
    
    def __init_fixed_set_cameras(self):

        #@ INIT INTRINSICS
        camera_lens = 35
        sensor_width = 32
        distances = 1.5
        focal_x = camera_lens * 2 / sensor_width
        focal_y = camera_lens * 2 / sensor_width
        principal_point = ((0,0),)

        #@ INIT EXTRINSICS
        x = torch.cos(self.azimuths_b64)*torch.cos(self.elevations_b64)
        y = torch.sin(self.azimuths_b64)*torch.cos(self.elevations_b64)
        z = torch.sin(self.elevations_b64)
        cam_pts = torch.stack([x,y,z],-1) * distances
        
        if self.up_vec == 'z':
            R, T = look_at_view_transform(eye=cam_pts, up=((0,0,1),))
        elif self.up_vec == '-z':
            R, T = look_at_view_transform(eye=cam_pts, up=((0,0,-1),))
        elif self.up_vec == 'y':
            R, T = look_at_view_transform(
                dist=distances,
                azim=self.azimuths_b64 * 180 / torch.pi + 90,
                elev=self.elevations_b64 * 180 / torch.pi,
                up=((0, 1, 0),),
            )
        else: raise NotImplementedError

        self.cameras_b64 = PerspectiveCameras(
                                R=R, 
                                T=T, 
                                focal_length = ((focal_x, focal_y),),
                                principal_point=principal_point,
                            )   
