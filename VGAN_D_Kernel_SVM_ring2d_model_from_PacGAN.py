import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR 
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import model_kernel_gaussian2d_from_PacGAN as model
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.svm import SVC
from data.ringDataset import ringDataset
import time

import os
import argparse

matplotlib.use('TkAgg')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_D_kernel')
parser.add_argument('--save_ims_dir', type=str, default='./saved_images')
parser.add_argument('--radius', type=float, default=1)

args = parser.parse_args()

save_dir = args.save_ims_dir
if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
radius = args.radius

# Noise
def noise(size):
    n = Variable(torch.randn(size, 2))
    if torch.cuda.is_available(): return n.cuda() 
    return n

generator = model.GeneratorNet()
if torch.cuda.is_available():
    generator = generator.cuda()
    
    
# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=5e-4)

scheduler_g = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=1)

# Number of steps to apply to the discriminator
g_steps = 5  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 2000

gamma = 0.2
gamma_scale_rate = 1.001


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def SVM_with_kernel(real_data, fake_data, gamma):
    
    N = len(real_data)
    data = torch.cat((real_data, fake_data), axis=0)
    
    y_ones = real_data_target(N)
    y_zeros = fake_data_target(N)
    y = torch.cat((y_ones, y_zeros), axis=0)
    
    X = data.cpu().detach().numpy()
    y = y.cpu().detach().numpy().reshape(-1)
    #gamma = 1/len(real_data[0])
    
    clf = SVC(gamma=gamma)
    clf.fit(X, y)
    alpha = clf.dual_coef_.reshape(-1)
    rho = clf.intercept_
    support_vecs = clf.support_vectors_
    
    return alpha, rho, support_vecs


def exp(x, y, data, variance, gamma):
    
    return variance * torch.exp(-gamma*((x-data[0])**2 + (y-data[1])**2))


def exp2(x, data, variance, gamma):
    
    return variance * torch.exp(-gamma * torch.norm(x - data, dim=1)**2)


def discriminator(x, y, alpha, rho, support_vecs, gamma):
    
    alpha = torch.from_numpy(alpha)
    rho = torch.from_numpy(rho)
    support_vecs = torch.from_numpy(support_vecs)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
        rho = rho.cuda()
        support_vecs = support_vecs.cuda()
    
    variance = 1
    #gamma = 1/(len(support_vecs[0]))
    D = 0
    for j in range(len(support_vecs)):
        D += alpha[j] * exp(x, y, support_vecs[j], variance, gamma)
            
    D += rho
    
    if torch.cuda.is_available():
        D = D.cuda()
    
    return D


def discriminator2(x, alpha, rho, support_vecs, gamma):
    
    alpha = torch.from_numpy(alpha)
    rho = torch.from_numpy(rho)
    support_vecs = torch.from_numpy(support_vecs)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
        rho = rho.cuda()
        support_vecs = support_vecs.cuda()
    
    variance = 1
    #gamma = 1/(len(support_vecs[0]))
    D = 0
    for j in range(len(support_vecs)):
        D += alpha[j] * exp2(x, support_vecs[j], variance, gamma)
            
    D += rho
    
    if torch.cuda.is_available():
        D = D.cuda()
    
    return D


def train_generator(optimizer, fake_data, alpha, rho, support_vecs, gamma):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    #start = time.time()
    
    #prediction = discriminator(fake_data, alpha, rho, support_vecs)
    prediction = discriminator2(fake_data, alpha, rho, support_vecs, gamma)
    prediction = prediction.reshape(-1, 1)
    if torch.cuda.is_available():
        prediction = prediction.cuda()
    #print('spent time for getting D is {}'.format(time.time()-start))
    # Calculate error and backpropagate
    y_ones = real_data_target(len(prediction))
    
    error = nn.ReLU()(1.0 - prediction).mean() #hinge loss
    #error = - prediction.mean()
    #error = nn.ReLU()(-prediction).mean()
    #start = time.time()
    error.backward()
    #print('spent time for backpropagation is {}'.format(time.time()-start))
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

for epoch in range(num_epochs):
    
    # Load data
    data = ringDataset(n=8, n_data=200, r=radius)
    # Create loader with data, so that we can iterate over it
    #batch_size=100
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    # Num batches
    num_batches = len(data_loader)
    
    for n_batch, (real_batch) in enumerate(tqdm(data_loader)):
        
        # 1. Train Discriminator
        if n_batch % g_steps == 0:
            if n_batch == 0 and epoch == 0:
                real_g_data = data.data[:g_steps*len(real_batch)]
                real_g_data = torch.from_numpy(real_g_data).type(torch.FloatTensor)
            if torch.cuda.is_available():
                real_g_data = real_g_data.cuda()
            fake_g_data = generator(noise(real_g_data.size(0))).detach()
            #start = time.time()
            alpha, rho, support_vecs = SVM_with_kernel(real_g_data, fake_g_data, gamma)
            #print('spent time for getting SVM is {}'.format(time.time()-start))
            real_g_data = torch.zeros(g_steps*len(real_batch), len(real_batch[0]))
        
        k = n_batch % g_steps
        real_data = real_batch.type(torch.FloatTensor)
        if torch.cuda.is_available(): 
            real_data = real_data.cuda()
        real_g_data[k*len(real_batch):(k+1)*len(real_batch)] = real_data

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_data.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data, alpha, rho, support_vecs, gamma)
        '''
        X, Y = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        Z = discriminator2(X_tensor, Y_tensor, alpha, rho, support_vecs)
        Z = Z.cpu().detach().numpy()
        real_data = real_data.cpu().detach().numpy()
        fake_data = fake_data.cpu().detach().numpy()
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_title('discriminator graph')
        ax.scatter(real_data[:,0], real_data[:,1], marker='*')
        plt.axis('equal')
        ax.scatter(fake_data[:,0], fake_data[:,1], marker='.')
        plt.axis('equal')
        
        plt.show()
        plt.close(fig)
        '''
    
    if epoch % 1 == 0:
        #start = time.time()
        X, Y = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            Y_tensor = Y_tensor.cuda()
        Z = discriminator(X_tensor, Y_tensor, alpha, rho, support_vecs, gamma)
        Z = Z.cpu().detach().numpy()
        real_t_data = data.data
        fake_t_data = generator(noise(500))
        fake_t_data = fake_t_data.cpu().detach().numpy()
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=1, fontsize=10)
        fig.suptitle('2d ring at epoch {} with gamma {}'.format(epoch, gamma))
        ax.scatter(real_t_data[:,0], real_t_data[:,1], marker='*')
        plt.axis('equal')
        ax.scatter(fake_t_data[:,0], fake_t_data[:,1], marker='.')
        plt.axis('equal')
        ax.set_xlim([-2*radius, 2*radius])
        ax.set_ylim([-2*radius, 2*radius])
        
        plt.savefig(os.path.join(save_dir, '{}.png'.format(str(epoch).zfill(3))), bbox_inches='tight')
        #plt.show()
        plt.close(fig)
        #print('spent time for drawing graph is {}'.format(time.time()-start))
        
        print('epoch # :', epoch, 'gen loss :', g_error.item())

    gamma *= gamma_scale_rate
    scheduler_g.step()
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(str(epoch).zfill(3))))

final_samples = generator(noise(2500)).cpu().detach().numpy()
#print('# ',(np.linalg.norm(final_samples, axis=1) < 3 * data.sig ** 0.5).sum())
quantitative_results = data.format_metric(final_samples)
with open(os.path.join(save_dir, 'quantitative.txt'), 'w') as f:
    for key, val in quantitative_results.items():
        print('{}: {}'.format(key, val))
        f.write('{}: {}\n'.format(key, val))