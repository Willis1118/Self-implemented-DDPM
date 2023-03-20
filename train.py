#!/bin/env python

import time
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.utils import save_image
from torch.utils.data import DataLoader

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

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    
    return logger

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    
    return arr
'''
    Below is the training script
'''
if __name__ == '__main__':

    torch.manual_seed(0)

    settings.init()

    logger = create_logger('experiments/results')

    logger.info("========== Loading Dataset ==========")

    dataset_train = ImageNet(
        root = settings.dir_path,
        split = 'train',
        transform = get_transform(),
    )

    data_loader = DataLoader(dataset_train, batch_size=settings.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device:{torch.cuda.current_device()}")

    model = Unet(
        dim=settings.image_size,
        init_dim=settings.image_size,
        use_convnext=False
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start = time.time()

    logger.info("========== Start Training ==========")

    for epoch in range(settings.epochs):
        for step, data in enumerate(data_loader):
            batch, _ = data
            batch = batch.to(device)

            optimizer.zero_grad()

            # following algo 1 line 3, uniformly sample timestep for each example in the batch
            t = torch.randint(0, settings.T, (batch.shape[0], ), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber") #--> smooth l1 loss

            if step % 100 == 0:
                elapsed = time.time() - start
                logger.info('Epoch: {}, Step: {}, Loss: {}, Stpes/Sec: {}'.format(epoch, step, loss.item(), step / elapsed))
            
            loss.backward()
            optimizer.step()


            # if step != 0 and step % settings.save_every == 0:
            #     milestone = step // settings.save_every
            #     batches = num_to_groups(4, batch.shape[0])
            #     all_imgs_list = list(map(lambda n: sample(model, settings.image_size, batch_size=n, channels=3), batches))
            #     all_imgs = torch.stack(all_imgs_list, dim=0)
            #     all_imgs = (all_imgs + 1) * 0.5
            #     save_image(all_imgs, str(f'experiments/results/sample-{milestone}'), nrow=6)



    

