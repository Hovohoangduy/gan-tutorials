import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

random.seed(1105)
torch.manual_seed(1105)
torch.use_deterministic_algorithms(True)

dataroot = 'data/celebs'
workers = 2 # number of worker for dataloader
batch_size = 128
image_size = 64
nc = 3 # number of channels in the images
nz = 10 # size of z latent vector (size of generator input)
ngf = 64 # size of feature maps in generator
ndf = 64 # size of feature maps in discriminator
num_epochs = 5
lr = 0.0002
beta1 = 0.5 # beta1 hyperparameter for Adam optimizers
ngpu = 1 # number of GPUs availiable

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cup")