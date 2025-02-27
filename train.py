import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data, lidar_bev_cam_correspondences

import pathlib
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
import random
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp

from diskcache import Cache

# Records error and tracebacks in case of failure
@record
def main():
    # Removed torch.cuda.empty_cache() as it's not needed for CPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=41, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--load_file', type=str, default=None, help='ckpt to load.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start with. Useful when continuing trainings via load_file.')
    parser.add_argument('--setting', type=str, default='all', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')
    #alter
    #parser.add_argument('--root_dir', type=str, default=r'/mnt/qb/geiger/kchitta31/datasets/carla/pami_v1_dataset_23_11', help='Root directory of your training data')
    parser.add_argument('--root_dir', type=str, default=r'D:/Lab 6/transfuser/team_code_transfuser/data/rr_dataset_23_11', help='Root directory of your training data')

    parser.add_argument('--schedule', type=int, default=1,
                        help='Whether to train with a learning rate schedule. 1 = True')
    parser.add_argument('--schedule_reduce_epoch_01', type=int, default=30,
                        help='Epoch at which to reduce the lr by a factor of 10 the first time. Only used with --schedule 1')
    parser.add_argument('--schedule_reduce_epoch_02', type=int, default=40,
                        help='Epoch at which to reduce the lr by a factor of 10 the second time. Only used with --schedule 1')
    parser.add_argument('--backbone', type=str, default='transFuser',
                        help='Which Fusion backbone to use. Options: transFuser, late_fusion, latentTF, geometric_fusion')
    parser.add_argument('--image_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the image branch. efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the lidar branch. Tested: efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--use_velocity', type=int, default=0,
                        help='Whether to use the velocity input. Currently only works with the TransFuser backbone. Expected values are 0:False, 1:True')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers used in the transfuser')
    parser.add_argument('--wp_only', type=int, default=0,
                        help='Valid values are 0, 1. 1 = using only the wp loss; 0= using all losses')
    parser.add_argument('--use_target_point_image', type=int, default=1,
                        help='Valid values are 0, 1. 1 = using target point in the LiDAR0; 0 = dont do it')
    parser.add_argument('--use_point_pillars', type=int, default=0,
                        help='Whether to use the point_pillar lidar encoder instead of voxelization. 0:False, 1:True')
    parser.add_argument('--parallel_training', type=int, default=1,
                        help='If this is true/1 you need to launch the train.py script with CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=123456780 --rdzv_backend=c10d train.py '
                             ' the code will be parallelized across GPUs. If set to false/0, you launch the script with python train.py and only 1 GPU will be used.')
    parser.add_argument('--val_every', type=int, default=5, help='At which epoch frequency to validate.')
    parser.add_argument('--no_bev_loss', type=int, default=0, help='If set to true the BEV loss will not be trained. 0: Train normally, 1: set training weight for BEV to 0')
    parser.add_argument('--sync_batch_norm', type=int, default=0, help='0: Compute batch norm for each GPU independently, 1: Synchronize Batch norms accross GPUs. Only use with --parallel_training 1')
    parser.add_argument('--zero_redundancy_optimizer', type=int, default=0, help='0: Normal AdamW Optimizer, 1: Use Zero Reduncdancy Optimizer to reduce memory footprint. Only use with --parallel_training 1')
    parser.add_argument('--use_disk_cache', type=int, default=0, help='0: Do not cache the dataset 1: Cache the dataset on the disk pointed to by the SCRATCH enironment variable. Useful if the dataset is stored on slow HDDs and can be temporarily stored on faster SSD storage.')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)
    parallel = False  # Force parallel training to False for CPU execution

    # Disk cache setup remains unchanged as it's unrelated to GPU/CPU
    if bool(args.use_disk_cache):
        shared_dict = Cache(size_limit=int(768 * 1024 ** 3))
    else:
        shared_dict = None

    # Disable DDP and set device to CPU
    rank = 0
    local_rank = 0
    world_size = 1
    device = torch.device('cpu')

    # Configure config
    config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
    config.use_target_point_image = bool(args.use_target_point_image)
    config.n_layer = args.n_layer
    config.use_point_pillars = bool(args.use_point_pillars)
    config.backbone = args.backbone
    if bool(args.no_bev_loss):
        index_bev = config.detailed_losses.index("loss_bev")
        config.detailed_losses_weights[index_bev] = 0.0

    # Create model and move to CPU
    model = LidarCenterNet(config, device, args.backbone, args.image_architecture, args.lidar_architecture, bool(args.use_velocity))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)

    # Data setup remains unchanged, just ensure pin_memory is False for CPU
    train_set = CARLA_Data(root=config.train_data, config=config, shared_dict=shared_dict)
    val_set = CARLA_Data(root=config.val_data, config=config, shared_dict=shared_dict)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())

    dataloader_train = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=False)
    dataloader_val = DataLoader(val_set, shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=False)

    # Create logdir
    if not os.path.isdir(args.logdir):
        print('Created dir:', args.logdir)
        os.makedirs(args.logdir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.logdir) if rank == 0 else None
    if rank == 0:
        with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    if args.load_file is not None:
        model.load_state_dict(torch.load(args.load_file, map_location=device))
        optimizer.load_state_dict(torch.load(args.load_file.replace("model_", "optimizer_"), map_location=device))

    trainer = Engine(model=model, optimizer=optimizer, dataloader_train=dataloader_train, dataloader_val=dataloader_val,
                     args=args, config=config, writer=writer, device=device, rank=rank, world_size=world_size,
                     parallel=parallel, cur_epoch=args.start_epoch)

    for epoch in range(trainer.cur_epoch, args.epochs):
        if (epoch == args.schedule_reduce_epoch_01 or epoch == args.schedule_reduce_epoch_02) and args.schedule == 1:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.1
            print("Reduce learning rate by factor 10 to:", new_lr)
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        trainer.train()

        if args.setting != 'all' and epoch % args.val_every == 0:
            trainer.validate()

        trainer.save()

class Engine(object):
    def __init__(self, model, optimizer, dataloader_train, dataloader_val, args, config, writer, device, rank=0, world_size=1, parallel=False, cur_epoch=0):
        self.cur_epoch = cur_epoch
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.args = args
        self.config = config
        self.writer = writer
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.parallel = parallel
        self.vis_save_path = self.args.logdir + r'/visualizations'
        if self.config.debug:
            pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

        self.detailed_losses = config.detailed_losses
        detailed_losses_weights = config.detailed_losses_weights if not self.args.wp_only else [1.0] + [0.0] * 10
        self.detailed_weights = {key: detailed_losses_weights[idx] for idx, key in enumerate(self.detailed_losses)}

    def load_data_compute_loss(self, data):
        # Move data to CPU
        rgb = data['rgb'].to(self.device, dtype=torch.float32)
        depth = data['depth'].to(self.device, dtype=torch.float32) if self.config.multitask else None
        semantic = data['semantic'].squeeze(1).to(self.device, dtype=torch.long) if self.config.multitask else None
        bev = data['bev'].to(self.device, dtype=torch.long)
        lidar = data['lidar_raw' if self.config.use_point_pillars else 'lidar'].to(self.device, dtype=torch.float32)
        num_points = data['num_points'].to(self.device, dtype=torch.int32) if self.config.use_point_pillars else None
        label = data['label'].to(self.device, dtype=torch.float32)
        ego_waypoint = data['ego_waypoint'].to(self.device, dtype=torch.float32)
        target_point = data['target_point'].to(self.device, dtype=torch.float32)
        target_point_image = data['target_point_image'].to(self.device, dtype=torch.float32)
        ego_vel = data['speed'].to(self.device, dtype=torch.float32).reshape(-1, 1)

        if self.args.backbone in ['transFuser', 'late_fusion', 'latentTF']:
            losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                                target_point_image=target_point_image, ego_vel=ego_vel, bev=bev,
                                label=label, save_path=self.vis_save_path, depth=depth,
                                semantic=semantic, num_points=num_points)
        elif self.args.backbone == 'geometric_fusion':
            bev_points = data['bev_points'].long().to(self.device)
            cam_points = data['cam_points'].long().to(self.device)
            losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                                target_point_image=target_point_image, ego_vel=ego_vel, bev=bev,
                                label=label, save_path=self.vis_save_path, depth=depth,
                                semantic=semantic, num_points=num_points, bev_points=bev_points,
                                cam_points=cam_points)
        else:
            raise ValueError("Unsupported backbone: {}".format(self.args.backbone))

        return losses

    def train(self):
        self.model.train()
        num_batches = 0
        loss_epoch = 0.0
        detailed_losses_epoch = {key: 0.0 for key in self.detailed_losses}
        self.cur_epoch += 1

        for data in tqdm(self.dataloader_train):
            self.optimizer.zero_grad()
            losses = self.load_data_compute_loss(data)
            loss = sum(self.detailed_weights[key] * value for key, value in losses.items())
            loss.backward()
            self.optimizer.step()
            num_batches += 1
            loss_epoch += loss.item()
            for key in losses:
                detailed_losses_epoch[key] += self.detailed_weights[key] * losses[key].item()

        self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')

    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch = {key: 0.0 for key in self.detailed_losses}

        for data in tqdm(self.dataloader_val):
            losses = self.load_data_compute_loss(data)
            loss = sum(self.detailed_weights[key] * value for key, value in losses.items())
            loss_epoch += loss.item()
            num_batches += 1
            for key in losses:
                detailed_val_losses_epoch[key] += self.detailed_weights[key] * losses[key].item()

        self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')

    def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix):
        loss_epoch /= num_batches
        for key in detailed_losses_epoch:
            detailed_losses_epoch[key] /= num_batches

        if self.rank == 0:
            self.writer.add_scalar(prefix + 'loss_total', loss_epoch, self.cur_epoch)
            for key, value in detailed_losses_epoch.items():
                self.writer.add_scalar(prefix + key, value, self.cur_epoch)

    def save(self):
        if self.rank == 0:
            model_path = os.path.join(self.args.logdir, f'model_{self.cur_epoch}.pth')
            optim_path = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch}.pth')
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optim_path)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    mp.set_start_method('fork')
    main()