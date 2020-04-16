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
import model_kernel_mnist as model
from sklearn.svm import SVC
import time

import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_D_kernel_mnist')
parser.add_argument('--reg', type=str, default='specnorm', help='[specnorm, wdecay]')
parser.add_argument('--wdecay', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=0)

parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

DATA_FOLDER = './mnist_data'

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
#batch_size=100
data_loader = torch.utils.data.DataLoader(data, batch_size=200, shuffle=True)
# Num batches
num_batches = len(data_loader)

    
def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)
    
# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda() 
    return n

generator = model.GeneratorNet()
if torch.cuda.is_available():
    generator.cuda()
    
    
# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
#loss = nn.BCELoss()

# Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 200


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


def SVM_with_kernel(real_data, fake_data):
    
    N = len(real_data)
    data = torch.cat((real_data, fake_data), axis=0)
    
    y_ones = real_data_target(N)
    y_zeros = fake_data_target(N)
    y = torch.cat((y_ones, y_zeros), axis=0)
    
    X = data.cpu().detach().numpy()
    y = y.cpu().detach().numpy().reshape(-1)
    #gamma = 1/(len(real_data[0])*X.var())
    
    clf = SVC(gamma='auto')
    clf.fit(X, y)
    alpha = clf.dual_coef_.reshape(-1)
    rho = clf.intercept_
    support_vecs = clf.support_vectors_
    
    return alpha, rho, support_vecs


def exp(x, data, variance, gamma):
    
    return variance * torch.exp(-gamma * torch.norm(x - data, dim=1)**2)


def discriminator(x, alpha, rho, support_vecs):
    
    N = len(support_vecs)
    alpha = torch.from_numpy(alpha)
    rho = torch.from_numpy(rho)
    support_vecs = torch.from_numpy(support_vecs)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
        rho = rho.cuda()
        support_vecs = support_vecs.cuda()
    
    variance = 1
    gamma = 1/(len(support_vecs[0]))
    #D = torch.zeros((len(x), 1))
    D = 0
    
    for j in range(N):
        D += alpha[j] * exp(x, support_vecs[j], variance, gamma)
    
    D += rho
        
    if torch.cuda.is_available():
        D = D.cuda()
        
    return D


def train_generator(optimizer, fake_data, alpha, rho, support_vecs):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    #start = time.time()
    prediction = discriminator(fake_data, alpha, rho, support_vecs)
    prediction = prediction.reshape(-1, 1)
    if torch.cuda.is_available():
        prediction = prediction.cuda()
    #print('spent time for getting D is {}'.format(time.time()-start))
    # Calculate error and backpropagate
    error = nn.ReLU()(1.0 - prediction).mean()
    #start = time.time()
    error.backward()
    #print('spent time for backpropagation is {}'.format(time.time()-start))
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


def evaluate(epoch):

    samples = generator(noise(args.batch_size)).cpu().data.numpy()[:64]
    samples = samples.reshape(64, 28, 28)


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='gray')

    if not os.path.exists('out_mnist_kernel/'):
        os.makedirs('out_mnist_kernel/')

    plt.savefig('out_mnist_kernel/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)


num_test_samples = 16
test_noise = noise(num_test_samples)


#logger = Logger(model_name='VGAN', data_name='MNIST')

for epoch in range(num_epochs):
    start = time.time()
    for n_batch, (real_batch,_) in enumerate(tqdm(data_loader)):
        
        # 1. Train Discriminator
        real_data = images_to_vectors(real_batch)
        if torch.cuda.is_available(): 
            real_data = real_data.cuda()
        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        #start = time.time()
        alpha, rho, support_vecs = SVM_with_kernel(real_data, fake_data)
        #print('spent time for SVM is {}'.format(time.time()-start))

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data, alpha, rho, support_vecs)
        

        # Display Progress
        if (n_batch) % 100 == 0:
            
            test_images = vectors_to_images(generator(test_noise)).data.cpu()
            print('batch # :', n_batch, 'gen loss :', g_error.item())
            
    #print(real_data[0])
    #print(fake_data1[0])
    evaluate(epoch)
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(str(epoch).zfill(3))))
    print('spent time for one epoch is {}'.format(time.time()-start))
