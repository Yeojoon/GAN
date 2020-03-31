import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import model_gaussian2d as model
#from gpytorch.kernels import RBFKernel, ScaleKernel
from data.gaussianGridDataset import gaussianGridDataset
import time

import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
#parser.add_argument('--batch_size', type=int, default=64)
#parser.add_argument('--lr', type=float, default=2e-4)
#parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_D_kernel')
#parser.add_argument('--reg', type=str, default='specnorm', help='[specnorm, wdecay]')
#parser.add_argument('--wdecay', type=float, default=1e-3)
#parser.add_argument('--epoch', type=int, default=0)

#parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

# Load data
data = gaussianGridDataset(n=5, n_data=100, sig=0.01)
# Create loader with data, so that we can iterate over it
#batch_size=100
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)
    
# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda() 
    return n

discriminator = model.DiscriminatorNet()
generator = model.GeneratorNet()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()
    
    
# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-3)

# Loss function
#loss = nn.BCELoss()

# Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 10000


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    #data = Variable(torch.ones(size, 1))
    data = torch.ones(size, 1)
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    #data = Variable(torch.zeros(size, 1))
    data = torch.zeros(size, 1)
    if torch.cuda.is_available(): return data.cuda()
    return data


def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    N = len(real_data)
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)

    # Calculate error and backpropagate
    error = (torch.dist(prediction_real, real_data_target(N))**2 + torch.norm(prediction_fake)) / (2*N)
    error.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error


def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    N = len(fake_data)
    optimizer.zero_grad()
    # Sample noise and generate fake data
    #start = time.time()
    prediction = discriminator(fake_data)
    #print('spent time for getting D is {}'.format(time.time()-start))
    # Calculate error and backpropagate
    error = torch.dist(prediction, real_data_target(N))**2 / (2*N)
    #start = time.time()
    error.backward()
    #print('spent time for backpropagation is {}'.format(time.time()-start))
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

for epoch in range(num_epochs):
    for n_batch, (real_batch) in enumerate(tqdm(data_loader)):

        # 1. Train Discriminator
        real_data = real_batch.type(torch.FloatTensor)
        if torch.cuda.is_available(): 
            real_data = real_data.cuda()
        
        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        d_error = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_data.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        
        '''
        real_data = real_data.cpu().detach().numpy()
        fake_data = fake_data.cpu().detach().numpy()
        fig = plt.figure()
        plt.scatter(real_data[:,0], real_data[:,1], marker='*')
        plt.axis('equal')
        plt.scatter(fake_data[:,0], fake_data[:,1], marker='.')
        plt.axis('equal')
        
        plt.show()
        plt.close(fig)
        '''
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(str(epoch).zfill(3))))
    
    if epoch % 50 == 0:
        real_t_data = data.data
        fake_t_data = generator(noise(500))
        fake_t_data = fake_t_data.cpu().detach().numpy()
        fig, ax = plt.subplots()
        fig.suptitle('2d gaussiangrid at epoch {}'.format(epoch))
        plt.scatter(real_t_data[:,0], real_t_data[:,1], marker='*')
        plt.axis('equal')
        plt.scatter(fake_t_data[:,0], fake_t_data[:,1], marker='.')
        plt.axis('equal')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        
        plt.show()
        plt.close(fig)
        
        print('epoch # :', epoch, 'disc loss :', d_error.item(), 'gen loss :', g_error.item())
        
