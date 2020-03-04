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
import model

import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
#parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_mnist10')
parser.add_argument('--reg', type=str, default='specnorm', help='[specnorm, wdecay]')
parser.add_argument('--wdecay', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=0)

parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

DATA_FOLDER = './mnist_data'

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
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

discriminator = model.DiscriminatorNet()
generator = model.GeneratorNet()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()
    
    
# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()

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


def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)

    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
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

    if not os.path.exists('out_mnist10/'):
        os.makedirs('out_mnist10/')

    plt.savefig('out_mnist10/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)


num_test_samples = 16
test_noise = noise(num_test_samples)


#logger = Logger(model_name='VGAN', data_name='MNIST')

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(tqdm(data_loader)):

        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        if torch.cuda.is_available(): 
            #print('cuda')
            real_data = real_data.cuda()
        # Generate fake data
        #print(real_data.size(0))
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        

        # Display Progress
        if (n_batch) % 100 == 0:
            #display.clear_output(True)
            # Display Images
            test_images = vectors_to_images(generator(test_noise)).data.cpu()
            print('batch # :', n_batch, ', disc loss :', d_error.item(), ', gen loss :', g_error.item(), ', D(x) :', d_pred_real.cpu().detach().numpy()[0], ', D(G(z)) :',                                      d_pred_fake.cpu().detach().numpy()[0])
            #logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            #logger.display_status(
            #    epoch, num_epochs, n_batch, num_batches,
            #    d_error, g_error, d_pred_real, d_pred_fake
            #)
        # Model Checkpoints
        #logger.save_models(generator, discriminator, epoch)
    #print(real_data[0])
    #print(fake_data[0])
    #evaluate(epoch)
    #torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(str(epoch).zfill(3))))
    #torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(str(epoch).zfill(3))))
