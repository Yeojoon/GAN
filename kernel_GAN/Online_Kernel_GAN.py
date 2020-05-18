import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR 
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from online_kernel_classifier import KernelClassifier
import model.model_kernel_gaussian2d_from_PacGAN as model_gaussian2d
import model.model_kernel_mnist as model_mnist_VGAN
import model.model_DCGAN as model_DCGAN
from quantitative_metric import mode_collapse_metric
from sklearn.svm import SVC
from data.gaussianGridDataset import gaussianGridDataset
import time

import os



class Online_Kernel_GAN(object):
    def __init__(self, kernel='gaussian', lr=5e-4, lr_gamma=1, gamma=0.5, gamma_ratio=1.0, lmbda=0.1, eta=0.05, budget=2048, coef0=0.0, degree=3, alpha=0.5, lossfn='logistic', g_steps=5, e_steps=1, num_epochs=1000, n_data=100, batch_size=500, img_size=28, use_gpu=True, data=gaussianGridDataset(n=5, n_data=100, sig=0.05), data_type='gaussian2dgrid', model_type='VGAN'):
        self.kernel = kernel
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.gamma = gamma
        self.gamma_ratio = gamma_ratio
        self.lmbda = lmbda
        self.eta = eta
        self.budget = budget
        self.coef0 = coef0
        self.degree = degree
        self.alpha = alpha
        self.lossfn = lossfn
        self.g_steps = g_steps
        self.e_steps = e_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_gpu = use_gpu
        
        self.data_config = {
            'num_grid':5,
            'n_data':n_data,
            'sigma':0.05
        }
        
        self.data = data
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        self.data_type = data_type
        self.model_type = model_type
        
        if self.data_type == 'gaussian2dgrid':
            self.z_size = 2
        else:
            self.z_size = 100
        
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        
        if self.data_type == 'gaussian2dgrid':
            self.dim = 2
        elif self.data_type == 'mnist':
            self.dim = self.img_size**2
        else:
            self.dim = 3 * self.img_size**2
        
        if self.data_type == 'gaussian2dgrid':
            self.generator = model_gaussian2d.GeneratorNet()
        elif self.data_type == 'mnist':
            if self.model_type == 'VGAN':
                self.generator = model_mnist_VGAN.GeneratorNet()
            elif self.model_type == 'DCGAN':
                self.generator = model_DCGAN.generator_mnist()
        elif self.data_type == 'svhn':
            if self.model_type == 'DCGAN':
                self.generator = model_DCGAN.generator_svhn(z_size=self.z_size)
            
        if self.model_type == 'DCGAN':
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        else:
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=self.lr_gamma)
        self.generator.to(self.device)
        
        self.clf = KernelClassifier(dim=self.dim, kernel=self.kernel, gamma=self.gamma, gamma_ratio=self.gamma_ratio, alpha=self.alpha, lmbda=self.lmbda, eta=self.eta, budget=self.budget, lr=self.lr, img_size=self.img_size, use_gpu=self.use_gpu, lossfn=self.lossfn, data_type=self.data_type)
        
        
    def images_to_vectors(self, images):
        return images.view(images.size(0), self.img_size**2)

    
    def vectors_to_images(self, vectors):
        return vectors.view(vectors.size(0), 1, self.img_size, self.img_size)
    
    
    def noise(self, size):
        if self.data_type == 'mnist' and self.model_type == 'DCGAN':
            n = torch.randn((size, self.z_size), device=self.device).view(-1, self.z_size, 1, 1)
        else:
            n = torch.randn((size, self.z_size), device=self.device)
        return n


    def real_data_target(self, size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = torch.ones(size, device=self.device)
        return data


    def fake_data_target(self, size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = -torch.ones(size, device=self.device)
        return data


    def make_dataset(self, X, y, batch_size=64):
 
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
        return loader


    def online_kernel(self, real_data, fake_data):

        N = len(real_data)
        data = torch.cat((real_data, fake_data), 0)

        y_true = self.real_data_target(N)
        y_fake = self.fake_data_target(N)
        y = torch.cat((y_true, y_fake), 0)

        train_dataset = self.make_dataset(data, y)

        self.clf.train(train_dataset, num_epochs=3)

    
    def exp(self, x, y, data):

        return torch.exp(-self.gamma*((x-data[0])**2 + (y-data[1])**2))
    
    
    def discriminator(self, x):
        
        D = self.clf.funceval(x)

        return D


    def discriminator_for_graph(self, x, y):

        alpha = self.clf.alphas.type(torch.FloatTensor)
        rho = self.clf.offset.type(torch.FloatTensor)
        key_vecs = self.clf.keypoints.type(torch.FloatTensor)
        if self.use_gpu and torch.cuda.is_available():
            alpha = alpha.cuda()
            rho = rho.cuda()
            key_vecs = key_vecs.cuda()

        D = 0
        for j in range(len(key_vecs)):
            D += alpha[j] * self.exp(x, y, key_vecs[j])

        D += rho

        if self.use_gpu and torch.cuda.is_available():
            D = D.cuda()

        return D


    def train_generator(self, optimizer, fake_data):
        # 2. Train Generator
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        #start = time.time()
        prediction = self.discriminator(fake_data)
        prediction = prediction.reshape(-1, 1)
        if self.use_gpu and torch.cuda.is_available():
            prediction = prediction.cuda()
        #print('spent time for getting D is {}'.format(time.time()-start))
        # Calculate error and backpropagate
        #error = - nn.LogSigmoid()(prediction).mean()
        error = nn.ReLU()(1.0 - prediction).mean() #hinge loss

        #start = time.time()
        error.backward()
        #print('spent time for backpropagation is {}'.format(time.time()-start))
        # Update weights with gradients
        optimizer.step()
        # Return error
        return error

    
    def evaluate_gaussian2d(self, epoch):
        
        X, Y = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        if self.use_gpu and torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            Y_tensor = Y_tensor.cuda()
        Z = self.discriminator_for_graph(X_tensor, Y_tensor)
        Z = Z.cpu().detach().numpy()
        real_t_data = self.data.data
        fake_t_data = self.generator(self.noise(500))
        fake_t_data = fake_t_data.cpu().detach().numpy()
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=1, fontsize=10)
        fig.suptitle('2d gaussiangrid at epoch {}'.format(epoch))
        ax.scatter(real_t_data[:,0], real_t_data[:,1], marker='*')
        plt.axis('equal')
        ax.scatter(fake_t_data[:,0], fake_t_data[:,1], marker='.')
        plt.axis('equal')
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])

        if not os.path.exists('out_image/'):
            os.makedirs('out_image/')
        if not os.path.exists('out_image/out_gaussian2d_online_kernel'):
            os.makedirs('out_image/out_gaussian2d_online_kernel')

        plt.savefig('out_image/out_gaussian2d_online_kernel/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        samples = self.generator(self.noise(2500)).cpu().detach().numpy()
        num_modes, num_high_qual_samples, mode_counter = mode_collapse_metric(samples, self.data_config['num_grid'], self.data_config['sigma'])

        print('Total number of generated modes is ', num_modes)
        print('The number of high quality samples among 2500 samples is', num_high_qual_samples)
        print('The mode dictionary is', mode_counter)

        
    def evaluate_image(self, epoch):

        samples = self.generator(self.noise(64)).cpu().data.numpy()
        if self.data_type == 'mnist':
            samples = samples.reshape(64, self.img_size, self.img_size)
        elif self.data_type == 'svhn':
            samples = samples.reshape(64, 3, self.img_size, self.img_size)

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if self.data_type == 'mnist':
                plt.imshow(sample, cmap='gray')
            else:
                sample = np.transpose(sample, (1, 2, 0))
                sample = ((sample +1)*255 / (2)).astype(np.uint8)
                plt.imshow(sample.reshape(self.img_size, self.img_size, 3))

        if not os.path.exists('out_image/'):
            os.makedirs('out_image/')
        if self.data_type == 'mnist':
            if not os.path.exists('out_image/out_mnist_online_kernel'):
                os.makedirs('out_image/out_mnist_online_kernel')

            plt.savefig('out_image/out_mnist_online_kernel/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        elif self.data_type == 'svhn':
            if not os.path.exists('out_image/out_svhn_online_kernel'):
                os.makedirs('out_image/out_svhn_online_kernel')

            plt.savefig('out_image/out_svhn_online_kernel/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        
        
    def train_GAN(self):  
            
        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            for n_batch, (real_batch) in enumerate(tqdm(self.data_loader)):
                if self.data_type is not 'gaussian2dgrid':
                    real_batch = real_batch[0] # real_batch is tuple for mnist, svhn, ... (data, target)
                    
                if self.data_type == 'gaussian2dgrid':
                    real_data = real_batch.type(torch.FloatTensor)
                elif self.data_type == 'mnist':
                    real_data = self.images_to_vectors(real_batch)
                else:
                    real_data = real_batch
                    
                if self.use_gpu and torch.cuda.is_available(): 
                    real_data = real_data.cuda()
                    
                if self.model_type == 'DCGAN' and self.data_type == 'mnist':
                    fake_data = self.images_to_vectors(self.generator(self.noise(real_data.size(0))).detach())
                else:
                    fake_data = self.generator(self.noise(real_data.size(0))).detach()
                    
                if self.data_type == 'svhn' and (epoch != 0 or n_batch != 0):
                    for i in range(self.e_steps):
                        fake_data = self.generator(self.noise(real_data.size(0))).detach()
                        e_error = self.clf.train_encoder(real_data, fake_data)
                #start = time.time()   
                #print(real_data.shape)
                #print(fake_data.shape)
                self.online_kernel(real_data, fake_data)
                #print('spent time for SVM is {}'.format(time.time() - start))
                # 2. Train Generator
                # Generate fake data
                for i in range(self.g_steps):
                    if self.model_type == 'DCGAN' and self.data_type == 'mnist':
                        fake_data = self.images_to_vectors(self.generator(self.noise(real_data.size(0))))
                    else:
                        fake_data = self.generator(self.noise(real_data.size(0)))
                    # Train G
                    g_error = self.train_generator(self.g_optimizer, fake_data)

            self.scheduler_g.step()
            if not os.path.exists('checkpoint/'):
                os.makedirs('checkpoint/')
            
            if self.data_type == 'gaussian2dgrid':
                checkpoint_folder = 'checkpoint/checkpoint_gaussian2dgrid_online_kernel'
            elif self.data_type == 'mnist':
                checkpoint_folder = 'checkpoint/checkpoint_mnist_online_kernel'
            elif self.data_type == 'svhn':
                checkpoint_folder = 'checkpoint/checkpoint_svhn'
                
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            torch.save(self.generator.state_dict(), os.path.join(checkpoint_folder, 'gen_{}'.format(str(epoch).zfill(3))))
            print('spent time for epoch {} is {}s'.format(epoch, time.time()-epoch_start))
            if self.data_type == 'svhn':
                print('epoch # :', epoch, 'gamma :', self.clf.gamma, 'gen loss :', g_error.item(), 'encoder loss :', e_error.item())
            else:
                print('epoch # :', epoch, 'gamma :', self.clf.gamma, 'gen loss :', g_error.item())
            
            if self.data_type == 'gaussian2dgrid':
                self.evaluate_gaussian2d(epoch)
            else:
                self.evaluate_image(epoch)
                
            if self.data_type == 'gaussian2dgrid':
                self.clf.gamma_update()
