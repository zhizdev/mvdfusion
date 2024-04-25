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

from utils.common_utils import rank_zero_print, AverageMeter, uncollate
from utils.data_sampler_utils import StatefulDistributedSampler
from utils.vis_utils import visualize_image, visualize_image_diffusion, visualize_image_depth_diffusion


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
    train_dataset = instantiate_from_config(config['dataset'])
    train_sampler = StatefulDistributedSampler(train_dataset,
                                               num_replicas=args.world_size,
                                               rank=rank,
                                               start_iter=local_step,
                                               batch_size=config['dataset']['scene_batch_size'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=config['dataset']['scene_batch_size'],
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True,
                                            sampler=train_sampler)
    rank_zero_print(rank, f'loaded dataset...', flush=True)
    assert config['dataset']['scene_batch_size'] == 1, f'does not support scene batch size greater than 1'

    #@ DEFINE TRAIN PARAMS
    max_epoch = config['trainer']['epochs']
    max_local_step = len(train_loader)
    loss_interval = config['saver']['loss_interval']
    print_interval = float(config['saver']['print_interval'])
    vis_interval = config['saver']['vis_interval']
    save_interval = config['saver']['save_interval']
    trainer_config = dict(config['trainer'])

    start_epoch = epoch
    if rank == 0:
        loss_list = []
        loss_running = AverageMeter()        
        step_start = time.time()
        first_local_step = True
        print('starting training...')
        print(f'training from epoch: {epoch} | local_step: {local_step} | global_step: {global_step}')

    #@ TRAINING LOOP
    for ep in range(start_epoch, max_epoch):

        #@ RESET DATA SAMPLER
        train_loader.sampler.set_epoch(ep, zero_start=False)
        if ep > start_epoch:
            local_step = 0
            train_loader.sampler.set_epoch(ep)

        #@ EPOCH LOOP
        for batch in train_loader:
            
            batch = uncollate(batch, to_device=f'cuda:{gpu}')

            #@ TRAINING STEP
            loss = model(batch, trainer_config)
            
            #@ BACKWARD PASS
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            #@ VISUALIZATION AND SAVING
            if rank == 0:

                #@ HANDLE LOSS
                loss_running.update(loss.item())
                if not first_local_step and global_step % loss_interval == 0:
                    loss_list.append(loss_running.avg)
                    loss_running.reset()

                #@ PRINT PROGRESS
                if not first_local_step and global_step % print_interval == 0:
                    nowtime =str(datetime.datetime.now()).split('.')[0]
                    step_time = time.time() - step_start
                    step_start = time.time()
                    print(f'{nowtime} | loss {loss:.2E} | epoch {ep}/{max_epoch} | step {local_step}/{max_local_step} | global {global_step} |', end='')
                    print(f'{print_interval/step_time:.02f}it/s | epoch eta {max_local_step/print_interval*step_time/60.0:.0f} min', flush=True)
                
                #@ HANDLE VISUALIZATION
                if global_step % vis_interval == 0:
                    visualize_image_depth_diffusion(config, model, batch, global_step=global_step, loss_list=loss_list, concat_input=config['saver']['concat_input'], overwrite_x_noisy=config['saver']['overwrite_x_noisy'])

                #@ HANDLE SAVING
                if not first_local_step and global_step % save_interval == 0:
                    save_model(config, model, optimizer, global_step=global_step, local_step=local_step, epoch=ep)

                if not first_local_step and global_step % (10*save_interval) == 0:
                    save_model(config, model, optimizer, global_step=global_step, local_step=local_step, epoch=ep, mod=f'last_{10*save_interval}')

                #@ UPDATE TRACKERS
                first_local_step = False
                global_step += 1
                local_step += 1

            

def load_model(gpu, config):

    #@ INIT MODEL CLASS
    model = instantiate_from_config(config['model']).cuda(gpu)
    optimizer = model.configure_optimizers(lr=config['trainer']['lr'])
    if gpu == 0:
        model._print_parameter_count()

    save_dir = config['saver']['exp_dir'] + config['saver']['ckpt_dir']
    ckpt_path = f'{save_dir}/latest.pt'

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


def save_model(config, model, optimizer, global_step, local_step, epoch, mod='latest'):

    #@ SAVE MODEL
    save_dir = config['saver']['exp_dir'] + config['saver']['ckpt_dir']
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            'local_step': local_step,
            'global_step': global_step,
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        f'{save_dir}/{mod}.pt'
    )


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
    parser.add_argument('-c', '--config', type=str, default='configs/mvd_train.yaml', metavar='S',
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