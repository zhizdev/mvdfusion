import os
import argparse
import imageio
import numpy as np
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import datetime
import time
from omegaconf import OmegaConf

import torch
from pytorch3d.renderer import PerspectiveCameras

import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning import seed_everything

from utils.common_utils import rank_zero_print, AverageMeter, uncollate, unnormalize, dict_to_device, split_list
from utils.data_sampler_utils import StatefulDistributedSampler
from utils.load_model import instantiate_from_config


def train(gpu, args):

    #@ INIT DISTRIBUTED PROCESS
    rank = args.nr * args.gpus + gpu
    print('spawning gpu rank', rank, 'out of', args.gpus, 'using', args.backend, flush=True)
    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    torch.cuda.set_device(gpu)

    #@ LOAD MODEL
    config = OmegaConf.load(str(args.config))
    model, optimizer, global_step, local_step, epoch = load_model(gpu, config)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    rank_zero_print(rank, f'loaded model...', flush=True)

    #@ LOAD DATASET
    config['dataset']['params']['stage'] = config['inference']['stage']
    train_dataset = instantiate_from_config(config['dataset'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=config['dataset']['scene_batch_size'],
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True,
                                            )
    rank_zero_print(rank, f'loaded dataset...', flush=True)
    assert config['dataset']['scene_batch_size'] == 1, f'does not support scene batch size greater than 1'

    #@ DEFINE TRAIN PARAMS
    inference_config = dict(config['inference'])

    #@ EPOCH LOOP
    model.eval()

    start_time = time.time()
    eta = 0.0
    eta_min = 0.0
    eta_hours = 0.0
    config['inference']['eval_num'] = config['inference'].get('eval_num', 20)
    config['inference']['eval_num'] = min(config['inference']['eval_num'], len(train_dataset))

    total_val_list = torch.arange(config['inference']['eval_num'])
    val_list = split_list(total_val_list, args.gpus)[gpu]
    print(f'gpu {gpu}: {val_list}')

    #@ 
    for bi, val_idx in enumerate(val_list):

        if bi > 0:
            elapsed_time = time.time() - start_time
            remaining_nums = len(val_list) - bi
            eta_rate = float(bi) / elapsed_time
            eta = remaining_nums / eta_rate
            eta_min = (eta / 60) % 60
            eta_hours = (eta // 3600)

        print(f'evaluating scene {val_idx} on gpu {gpu} | scene: {bi}/{len(val_list)} | eta: {eta_hours} hours {eta_min} mins')
        
        batch = train_dataset.__getitem__(val_idx)
        batch = dict_to_device(batch, to_device=f'cuda:{gpu}')

        #@ INFERENCE
        with torch.no_grad():
            model_outputs = model.module.sample(batch, inference_config, cfg_scale=inference_config['cfg_scale'], return_input=True, depth=True, verbose=True)

        if len(model_outputs) == 3:
            model_output, batch_latents, input_latents = model_outputs
        elif len(model_outputs) == 5:
             model_output, batch_latents, input_latents, batch_cameras, intermediates = model_outputs

        pred_rgb = model.module.decode(model_output[:,:4,...])
        input_rgb = model.module.decode(input_latents[:,:4,...])
        gt_rgb = model.module.decode(batch_latents[:,:4,...])

        pred_rgb = rearrange(pred_rgb, 'b c h w -> b h w c').detach().cpu().numpy()
        input_rgb = rearrange(input_rgb, 'b c h w -> b h w c').detach().cpu().numpy()
        gt_rgb = rearrange(gt_rgb, 'b c h w -> b h w c').detach().cpu().numpy()

        #@ SAVE LOCATIONS
        save_dir = config['saver']['exp_dir'] + config['inference']['vis_dir']
        os.makedirs(save_dir, exist_ok=True)

        n_frames = inference_config['train_batch_size']
        jpg_path = f'{save_dir}/{global_step:07d}_eval_{val_idx:03d}_n{n_frames}.jpg'
        gif_path = f'{save_dir}/{global_step:07d}_eval_{val_idx:03d}_n{n_frames}.gif'
        
        vis_pred = pred_rgb
        vis_pred_wide = np.hstack(vis_pred)
        vis_gt_wide = np.hstack(gt_rgb)

        #@ SAVE IMAGE
        vis = vis_pred_wide
        imageio.imwrite(jpg_path, (vis*255).astype(np.uint8))

        #@ SAVE GIF
        with imageio.get_writer(gif_path, mode='I', duration=0.2) as writer:
                for j in range(len(vis_pred)):
                    vis_single = np.hstack((gt_rgb[j], vis_pred[j]))
                    writer.append_data(((vis_single)*255).astype(np.uint8))
        print('saved video', gif_path)

        pred_depth = unnormalize(model_output[:,4:,...])
        input_depth = unnormalize(input_latents[:,4:,...])
        gt_depth = unnormalize(batch_latents[:,4:,...])

        pred_depth = pred_depth.clip(0.0, 1.0)

        pred_depth = rearrange(pred_depth, 'b c h w -> b h w c').expand(-1,-1,-1,3).detach().cpu().numpy()
        input_depth = rearrange(input_depth, 'b c h w -> b h w c').expand(-1,-1,-1,3).detach().cpu().numpy()
        gt_depth = rearrange(gt_depth, 'b c h w -> b h w c').expand(-1,-1,-1,3).detach().cpu().numpy()

        vis_pred_depth = np.hstack(pred_depth)
        vis_input_depth = np.hstack(input_depth)
        vis_gt_depth = np.hstack(gt_depth)
        vis_depth = np.hstack((vis_input_depth, vis_pred_depth))
        imageio.imwrite(jpg_path.replace('.jpg', '_depth.png'), (vis_depth*255).astype(np.uint8))

        depth_npz_path = jpg_path.replace('.jpg', '_depth.npy')
        with open(depth_npz_path, 'wb') as fp:
            np.save(fp, vis_depth)

        gif_path = jpg_path.replace('.jpg', '_depth.gif')
        with imageio.get_writer(gif_path, mode='I', duration=0.2) as writer:
            for j in range(len(vis_pred)):
                vis_single = np.hstack((pred_depth[j]))
                writer.append_data(((vis_single)*255).astype(np.uint8))
            

def load_model(gpu, config):

    #@ INIT MODEL CLASS
    model = instantiate_from_config(config['model']).cuda(gpu)
    optimizer = model.configure_optimizers(lr=config['trainer']['lr'])
    if gpu == 0:
        model._print_parameter_count()

    ckpt_path = config['saver']['ckpt_path']

    #@ RESUME MODEL
    if os.path.exists(ckpt_path):
        if gpu == 0:
            print('***\n***Loading model from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt['global_step']
        epoch = ckpt['epoch']
        local_step = ckpt['local_step']

    #@ INIT NEW MODEL
    else:
        if gpu == 0:
            print('***\n***WARNING!\n***Initializing model from scratch')
        global_step = 0
        local_step = 0
        epoch = 0

    return model, optimizer, global_step, local_step, epoch


def main():

    #@ HANDLE TRAINING TIME ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-p', '--port', default=1, type=int, metavar='N',
                        help='last digit of port (default: 1234[1])')
    parser.add_argument('-b', '--backend', type=str, default='nccl', metavar='s',
                        help='nccl')
    parser.add_argument('-c', '--config', type=str, default='configs/mvd_gso.yaml', metavar='S',
                        help='config file')
    args = parser.parse_args()

    #@ INIT DDP PORT
    args.world_size = args.gpus * args.nodes
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'1234{args.port}'
    print('using port', f'1234{args.port}')

    #@ SPAWN DDP PROCESSES
    mp.spawn(train, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()