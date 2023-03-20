from inspect import isfunction
from torch import nn

def exists(x):
    '''
        see if x is None or not
    '''
    return x is not None

def default(val, d):
    '''
        return val if val exists
        else return d
    '''
    if exists(val):
        return val
    return d() if isfunction(d) else d

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

