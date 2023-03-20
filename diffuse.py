import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
import requests
from tqdm.auto import tqdm

import settings

'''
    Below we define some variance schedulings
'''

def consine_beta_schedule(t, s=0.008):
    steps = t + 1

    # tensor in [0, t], total t+1
    # representing all timesteps up to T
    x = torch.linspace(0, t, steps)

    alphas_cumprod = torch.cos( ( ( x / t ) + s ) / ( 1 + s ) * torch.pi * 0.5)
    alphas_cumprod = alphas_cumprod ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # reduce all numbers outside of [0.0001, 0.9999] to the lower and upper bound, respectively
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(t):

    beta_start = 0.0001
    beta_end = 0.02

    return torch.linspace(beta_start, beta_end, t)

def quadratic_beta_schedule(t):

    beta_start = 0.0001
    beta_end = 0.02

    return torch.linspace(beta_start**0.5, beta_end**0.5, t) ** 2

def sigmoid_beta_schedule(t):

    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, t) # --> interval length is adjustable; endpoints is for betas to approach to 0 and 1 respectively

    # unnormalize
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def extract(a, t, x_shape):
    '''
        Given [alpha_1,...,alpha_t], extract the proper variance at time t
    '''
    batch_size = t.shape[0]

    # gather a based on index of t --> should be a tensor of shape (batch, 1)
    out = a.gather(-1, t.cpu())

    return out.reshape(batch_size, *( (1, ) * (len(x_shape) - 1) ) ).to(t.device)

'''
    Below we define sampling process following Langevin Dynamics
'''

def get_transform():
    return Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(), # turn into numpy array of shape HWC, divided by 255
        Lambda(lambda t: (t * 2) - 1),
    ])

def get_reverse_transform():
    return Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)), # (C, H ,W) --> (H, W, C)
        Lambda(lambda t: t * 255. ),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

def get_alphas(betas):
    return 1. - betas

def get_alphas_cumprod(alphas):
    return torch.cumprod(alphas, axis=0)

def get_alphas_cumprod_prev(alphas):
    '''
        Get alpha cumprod for the previous time step
    '''

    return F.pad(get_alphas_cumprod(alphas)[:-1], (1,0), value=1.0)

def get_posterior_params(alphas):
    '''
        Refer to the paper
    '''

    return {
        "mean": torch.sqrt(get_alphas_cumprod(alphas)),
        "minus_mean": torch.sqrt(1. - get_alphas_cumprod(alphas)),
        "variance": ( 1. - alphas ) * (1. - get_alphas_cumprod_prev(alphas)) / (1. - get_alphas_cumprod(alphas))
    }

def q_sample(x_start, t, noise=None):
    '''
        Forward diffusion process
    '''

    if noise is None: # pure standard Gaussian noise
        noise = torch.randn_like(x_start)
    
    alphas = get_alphas(linear_beta_schedule(settings.T))
    posterior = get_posterior_params(alphas)

    sqrt_alphas_cumprod_t = extract(
        posterior["mean"],
        t,
        x_start.shape,
    )
    sqrt_minus_alphas_cumprod = extract(
        posterior["minus_mean"],
        t,
        x_start.shape,
    )

    # following the reparametrization mentioned in the paper
    # where noise is standard Gaussian noise
    return sqrt_alphas_cumprod_t * x_start + sqrt_minus_alphas_cumprod * noise 

@torch.no_grad()
def p_sample(model, x, t, t_index):
    '''
        Reverse process
    '''
    betas = linear_beta_schedule(settings.T)
    betas_t = extract(betas, t, x.shape)
    
    sqrt_minus_alphas_cumprod_t = extract(
        get_posterior_params( get_alphas(betas) )["minus_mean"],
        t,
        x.shape
    )

    sqrt_recip_alphas_t = extract(
        torch.sqrt(1.0 / get_alphas(betas) ),
        t,
        x.shape
    )

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_var_t = extract(
            get_posterior_params( get_alphas(betas) )["variance"],
            t,
            x.shape
        )
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]

    # start from Gaussian noise
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, settings.T)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img)
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def get_noisy_image(x_start, t):
    # add noise
    x_noise = q_sample(x_start, t=t)

    # turn back to PIL image
    noisy_image = get_reverse_transform()(x_noise.squeeze())

    return noisy_image

def plot_noisy_img(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    # save figure
    plt.savefig('img_seq.png')
    plt.tight_layout()

if __name__ == '__main__':
    settings.init()
    torch.manual_seed(0)

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    image_size = 256

    transform = get_transform()
    reverse_transform = get_reverse_transform()

    x_start = transform(image).unsqueeze(0)

    assert x_start.shape == (1, 3, image_size, image_size)

    plot_noisy_img([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])
