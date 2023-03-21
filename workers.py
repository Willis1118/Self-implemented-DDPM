#!/bin/env python

import time
import logging
import argparse
import os
import random
from glob import glob
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import settings
from diffuse import plot_noisy_img, get_noisy_image, q_sample, sample
from model import Unet

def get_transform():
    return transforms.Compose([
        transforms.Resize(settings.image_size),
        transforms.CenterCrop(settings.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

def p_losses(model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    
    return arr

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def set_random_seed(random_seed=0):
    '''
        Set random seed for reproducibility
    '''
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

if __name__ == "__main__":

    settings.init()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=settings.epochs)
    parser.add_argument('--batch_size', type=int, default=settings.batch_size)
    parser.add_argument('--image_size', type=int, default=settings.image_size)
    parser.add_argument('--sample', action="store_true") # --> default false
    
    argv = parser.parse_args()

    epochs = argv.epochs
    batch_size = argv.batch_size
    image_size = argv.image_size
    sample_flag = argv.sample

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    assert batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    set_random_seed(rank)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed = {rank}, world_size={dist.get_world_size()}.")

    sample_dir = ""

    if rank == 0:
        os.makedirs('experiments/datapoints', exist_ok=True)
        experiment_index = len(glob(f"experiments/datapoints/*"))
        experiment_dir = f"experiments/datapoints/{experiment_index:3d}-DDPM"
        os.makedirs(experiment_dir, exist_ok=True)

        sample_index = len(glob(f"experiments/results/*"))
        sample_dir = f"experiments/results/{sample_index:3d}-DDPM"
        os.makedirs(sample_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f"learning_rate={settings.lr}, sampling_steps={settings.T}")
    else:
        logger = create_logger(None)
    
    model = Unet(
        dim=image_size,
        init_dim=image_size,
        use_convnext=False
    )

    model = DDP(model.to(device), device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)

    # ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # requires_grad(ema, False)

    dataset = ImageNet(
        root = settings.dir_path,
        split = 'train',
        transform = get_transform(),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=0
    )

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=dist.get_world_size(),
        pin_memory=True,
        drop_last=True
    )

    logger.info("========== Start Training ==========")

    # update_ema(ema, model.module, decay=0)
    model.train()
    # ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for step, data in enumerate(loader):
            batch, _ = data
            batch = batch.to(device)

            optimizer.zero_grad()

            # following algo 1 line 3, uniformly sample timestep for each example in the batch
            t = torch.randint(0, settings.T, (batch.shape[0], ), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber") #--> smooth l1 loss

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            loss.backward()
            optimizer.step()
            # update_ema(ema, model.module)

            if train_steps % 100 == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # sync over all proc
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            if sample_flag == True and train_steps % settings.save_every == 0 and rank == 0:

                logger.info("========== Sampling ==========")
                milestone = train_steps // settings.save_every
                all_imgs_list = sample(model, image_size, batch_size=batch_size, channels=3)
                all_imgs_list = (all_imgs_list[-1][:64] + 1) * 0.5
                save_image(all_imgs_list, str(f'{sample_dir}/sample-{milestone}.png'), nrow=8)
                logger.info("========== Sampling Done ==========")

    model.eval()

    logger.info("Done!")
    dist.destroy_process_group()

    

