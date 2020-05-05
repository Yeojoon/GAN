import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR 
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import model.model_kernel_gaussian2d_from_PacGAN as model_gaussian2d
import model.model_kernel_mnist as model_mnist_VGAN
import model.model_DCGAN as model_mnist_DCGAN
from quantitative_metric import mode_collapse_metric
from sklearn.svm import SVC
from data.gaussianGridDataset import gaussianGridDataset
import time

import os



class Kernel_SVM_GAN(object):
    def __init__(self, kernel='gaussian', lr=5e-4, lr_gamma=1, gamma=0.5, gamma_ratio=1.0, coef0=0.0, degree=3, g_steps=5, num_epochs=1000, n_data=100, batch_size=100, img_size=28, use_gpu=True, data=gaussianGridDataset(n=5, n_data=100, sig=0.05), data_type='gaussian2dgrid', model_type='VGAN'):
        self.kernel = kernel
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.gamma = gamma
        self.gamma_ratio = gamma_ratio
        self.coef0 = coef0
        self.degree = degree
        self.g_steps = g_steps
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
        
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        
        if self.data_type == 'gaussian2dgrid':
            self.generator = model_gaussian2d.GeneratorNet()
        elif self.data_type == 'mnist':
            if self.model_type == 'VGAN':
                self.generator = model_mnist_VGAN.GeneratorNet()
            elif self.model_type == 'DCGAN':
                self.generator = model_mnist_DCGAN.generator()
            
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=self.lr_gamma)
        
        self.generator.to(self.device)
    
    
    def images_to_vectors(self, images):
        return images.view(images.size(0), self.img_size**2)

    
    def vectors_to_images(self, vectors):
        return vectors.view(vectors.size(0), 1, self.img_size, self.img_size)
    
    
    def noise(self, size):
        if self.data_type == 'gaussian2dgrid':
            n = torch.randn((size, 2), device=self.device)
        elif self.data_type == 'mnist':
            if self.model_type == 'VGAN':
                n = torch.randn((size, 100), device=self.device)
            elif self.model_type == 'DCGAN':
                n = torch.randn((size, 100), device=self.device).view(-1, 100, 1, 1)
        return n


    def real_data_target(self, size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = torch.ones((size, 1), device=self.device)
        return data


    def fake_data_target(self, size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = torch.zeros((size, 1), device=self.device)
        return data


    def SVM_with_kernel(self, real_data, fake_data):

        N = len(real_data)
        data = torch.cat((real_data, fake_data), 0)

        y_ones = self.real_data_target(N)
        y_zeros = self.fake_data_target(N)
        y = torch.cat((y_ones, y_zeros), 0)

        X = data.cpu().detach().numpy()
        y = y.cpu().detach().numpy().reshape(-1)
        
        if self.kernel == 'gaussian':
            clf = SVC(gamma=self.gamma)
        elif self.kernel == 'poly':
            clf = SVC(gamma=self.gamma, kernel='poly')
            
        clf.fit(X, y)
        alpha = clf.dual_coef_.reshape(-1)
        rho = clf.intercept_
        support_vecs = clf.support_vectors_

        return alpha, rho, support_vecs

    
    def kernel_func(self, X, Z):
        if self.kernel == 'gaussian':
            return self.kernel_gaussian(X, Z)
        elif self.kernel == 'poly':
            return self.kernel_poly(X, Z)
        else:
            raise Exception('No kernel specified')
    

    def kernel_gaussian(self, X, Z):

        Xnorms = (X*X).sum(dim=1)
        Znorms = (Z*Z).sum(dim=1)
        onesn = torch.ones(X.shape[0], device=self.device)
        onesk = torch.ones(Z.shape[0], device=self.device)
        M = torch.ger(Xnorms,onesk) + torch.ger(onesn,Znorms) - 2 * (X @ Z.T)

        return torch.exp(-self.gamma*M)

    
    def kernel_poly(self, X, Z):
        
        return (self.gamma * X @ Z.T + self.coef0)**self.degree
    

    def exp(self, x, y, data):

        return torch.exp(-self.gamma*((x-data[0])**2 + (y-data[1])**2))
    
    
    def discriminator(self, x, alpha, rho, support_vecs):

        alpha = torch.from_numpy(alpha)
        rho = torch.from_numpy(rho)
        support_vecs = torch.from_numpy(support_vecs)
        if self.use_gpu and torch.cuda.is_available():
            alpha = alpha.type(torch.FloatTensor).cuda()
            rho = rho.type(torch.FloatTensor).cuda()
            support_vecs = support_vecs.type(torch.FloatTensor).cuda()

        D = self.kernel_func(x, support_vecs) @ alpha + rho

        return D


    def discriminator_for_graph(self, x, y, alpha, rho, support_vecs):

        alpha = torch.from_numpy(alpha)
        rho = torch.from_numpy(rho)
        support_vecs = torch.from_numpy(support_vecs)
        if self.use_gpu and torch.cuda.is_available():
            alpha = alpha.type(torch.FloatTensor).cuda()
            rho = rho.type(torch.FloatTensor).cuda()
            support_vecs = support_vecs.type(torch.FloatTensor).cuda()

        D = 0
        for j in range(len(support_vecs)):
            D += alpha[j] * self.exp(x, y, support_vecs[j])

        D += rho

        if torch.cuda.is_available():
            D = D.cuda()

        return D


    def train_generator(self, optimizer, fake_data, alpha, rho, support_vecs):
        # 2. Train Generator
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        #start = time.time()
        prediction = self.discriminator(fake_data, alpha, rho, support_vecs)
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

    
    def evaluate_gaussian2d(self, alpha, rho, support_vecs, epoch):
        
        X, Y = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        if self.use_gpu and torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            Y_tensor = Y_tensor.cuda()
        Z = self.discriminator_for_graph(X_tensor, Y_tensor, alpha, rho, support_vecs)
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
        if not os.path.exists('out_image/out_gaussian2d'):
            os.makedirs('out_image/out_gaussian2d')

        plt.savefig('out_image/out_gaussian2d/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        samples = self.generator(self.noise(2500)).cpu().detach().numpy()
        num_modes, num_high_qual_samples, mode_counter = mode_collapse_metric(samples, self.data_config['num_grid'], self.data_config['sigma'])

        print('Total number of generated modes is ', num_modes)
        print('The number of high quality samples among 2500 samples is', num_high_qual_samples)
        print('The mode dictionary is', mode_counter)

        
    def evaluate_mnist(self, epoch):

        samples = self.generator(self.noise(64)).cpu().data.numpy()
        samples = samples.reshape(64, self.img_size, self.img_size)

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

        if not os.path.exists('out_image/'):
            os.makedirs('out_image/')
        if not os.path.exists('out_image/out_mnist'):
            os.makedirs('out_image/out_mnist')

        plt.savefig('out_image/out_mnist/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        
        
    def train_GAN(self):  
            
        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            for n_batch, (real_batch) in enumerate(tqdm(self.data_loader)):
                if self.data_type == 'mnist':
                    real_batch = real_batch[0] # real_batch is tuple for mnist (data, target)
                # 1. Train Discriminator
                if n_batch % self.g_steps == 0:
                    if n_batch == 0 and epoch == 0:
                        if self.data_type == 'gaussian2dgrid':
                            real_g_data = self.data.data[:self.g_steps*len(real_batch)]
                            real_g_data = torch.from_numpy(real_g_data).type(torch.FloatTensor)
                        elif self.data_type == 'mnist':
                            real_g_data = self.images_to_vectors(real_batch)
                    if self.use_gpu and torch.cuda.is_available():
                        real_g_data = real_g_data.cuda()
                    
                    if self.model_type == 'DCGAN':
                        fake_g_data = self.images_to_vectors(self.generator(self.noise(real_g_data.size(0))).detach())
                    else:
                        fake_g_data = self.generator(self.noise(real_g_data.size(0))).detach()
                    #start = time.time()
                    alpha, rho, support_vecs = self.SVM_with_kernel(real_g_data, fake_g_data)
                    #print('spent time for getting SVM is {}'.format(time.time()-start))
                    real_g_data = torch.zeros((self.g_steps*len(real_batch), len(real_g_data[0])), device=self.device)

                k = n_batch % self.g_steps
                if self.data_type == 'gaussian2dgrid':
                    real_data = real_batch.type(torch.FloatTensor)
                elif self.data_type == 'mnist':
                    real_data = self.images_to_vectors(real_batch)
                    
                if self.use_gpu and torch.cuda.is_available(): 
                    real_data = real_data.cuda()
                real_g_data[k*len(real_batch):(k+1)*len(real_batch)] = real_data

                # 2. Train Generator
                # Generate fake data
                if self.model_type == 'DCGAN':
                    fake_data = self.images_to_vectors(self.generator(self.noise(real_data.size(0))))
                else:
                    fake_data = self.generator(self.noise(real_data.size(0)))
                # Train G
                g_error = self.train_generator(self.g_optimizer, fake_data, alpha, rho, support_vecs)

            self.scheduler_g.step()
            if not os.path.exists('checkpoint/'):
                os.makedirs('checkpoint/')
            
            if self.data_type == 'gaussian2dgrid':
                checkpoint_folder = 'checkpoint/checkpoint_gaussian2dgrid'
            elif self.data_type == 'mnist':
                checkpoint_folder = 'checkpoint/checkpoint_mnist'
                
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            torch.save(self.generator.state_dict(), os.path.join(checkpoint_folder, 'gen_{}'.format(str(epoch).zfill(3))))
            print('spent time for epoch {} is {}s'.format(epoch, time.time()-epoch_start))
            print('epoch # :', epoch, 'gamma :', self.gamma, 'gen loss :', g_error.item())
            
            if self.data_type == 'gaussian2dgrid':
                self.evaluate_gaussian2d(alpha, rho, support_vecs, epoch)
            elif self.data_type == 'mnist':
                self.evaluate_mnist(epoch)
                
            self.gamma *= self.gamma_ratio
