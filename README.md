# DDPM 

In this repo, we will be using the traditional U-Net structure with self-attention between sub-layers to traing DDPM on imagenet.

We will be taking the *predicting noise* interpretation for our denoising model implementation: given time step $t$ and image state $x_t$, our model will predict the noise added to image at the certain time step. 

The sampling step is implemented analoguously to Lanvegin Dynamics. 

## Positional Embeddings
Following **Attention is All You Need**. Denote position(in our case time $t$) by $p$, and dimension by $i$, the positional encoding is as follows: $$
\begin{align}
PE_{(p,\ 2i)} &= \sin(\frac{p}{10000^{2i/d_{model}}})\\
PE_{(p,\ 2i+1)} &= \cos(\frac{p}{10000^{2i/d_{model}}})
\end{align}
$$
According to the paper, this function is chosen due to the fact that for any fixed offset $k$, $PE_{(p+k)}$ can be expressed as a linear function of $PE_{p}$ (decomposition of trignometry). 

This is implemented in `model.py/PositionalEmbeddings`.

## Model Structure
Specified in `model.py`

## Variance scheduling
For the forward diffusion process, we used fixed variance.

In `diffuse.py` we defined several different variance schedulings with time $t$.

We furthermore follow the setting in paper of $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$

Other variance scheduling methods are free to explore.

## Sampling

To be implemented

## Distributed Training

By running

```
torchrun --nnodes=1 --nproc_per_node=N workers.py --batch_size <batch_size> --image_size <image_size> --epochs <epochs> <--sample>
```
where `N` is the number of GPU to train on, `<--sample>` is the flag to trigger sampling process with default step gap of 5000. 

## Model loading

To be implemented

## VAE latent learning

To be implemented


