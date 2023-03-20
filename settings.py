'''
    Global variable / hyperparameters
'''

def init():
    global T
    global image_size
    global batch_size
    global dir_path
    global epochs
    global save_every
    global lr

    T = 200
    image_size = 128
    batch_size = 64
    dir_path = '/vast/work/public/ml-datasets/imagenet'
    epochs = 10
    save_every = 5000
    lr = 5e-4

