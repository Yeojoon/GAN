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
import model.model_GAN_in_codespace as model_GAN_in_codespace
from lenet import LeNet5
from model.model_autoencoder import autoencoder, autoencoder_cifar10
from data.SyntheticDataset import gaussianGridDataset, ringDataset, circleDataset
from utils import generate_imgs
from score.score import get_inception_and_fid_score
import time

from IPython.display import clear_output


import os



class Online_Kernel_GAN(object):
    def __init__(self, kernel='gaussian', lr=5e-4, lr_gamma=1, lr_schedule=None, lr_IKL=None, lr_IKL_schedule=None, gamma=0.5, gamma_ratio=1.0, lmbda=0.1, lmbda_IKL=0.0, var_IKL=0.0, eta=0.05, budget=2048, coef0=0.0, degree=3, alpha=0.5, lossfn='logistic', g_steps=5, e_steps=1, IKL_steps=1,  num_epochs=1000, batch_size=500, img_size=28, code_dim=None, use_gpu=True, data=gaussianGridDataset(n=5, n_data=100, sig=0.05), data_type='gaussian2dgrid', model_type='VGAN'):
        self.kernel = kernel
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_schedule = lr_schedule
        self.lr_IKL = lr_IKL
        self.lr_IKL_schedule = lr_IKL_schedule
        self.gamma = gamma
        self.gamma_ratio = gamma_ratio
        self.lmbda = lmbda
        self.lmbda_IKL = lmbda_IKL
        self.var_IKL = var_IKL
        self.eta = eta
        self.budget = budget
        self.coef0 = coef0
        self.degree = degree
        self.alpha = alpha
        self.lossfn = lossfn
        self.g_steps = g_steps
        self.e_steps = e_steps
        self.IKL_steps = IKL_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.code_dim = code_dim
        self.use_gpu = use_gpu

        self.discriminator_reprs = []
                
        self.data = data
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        self.data_type = data_type
        self.model_type = model_type
        
        if self.data_type == 'gaussian2dgrid':
            self.z_size = 2
            self.division = 20
            self.epoch_arr = []
            self.reverse_kl = []
            if self.data.name == 'gmm2d':
                self.num_samples = 5000
            else:
                self.num_samples = 2500
        else:
            if self.model_type == 'GAN with AE':
                self.z_size = 10
            else:
                self.z_size = 100
            self.division = 1
        
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        
        if self.data_type == 'gaussian2dgrid':
            self.dim = 2
        else:
            if self.model_type == 'GAN with AE' or self.model_type == 'DCGAN':
                self.dim = code_dim
            else:
                if self.data_type == 'mnist':
                    self.dim = self.img_size**2
        
        if self.model_type == 'GAN with AE':
            if self.data_type == 'mnist':
                autoencoder_model_path = './checkpoint_autoencoder/autoencoder_dim{}.pth'.format(self.code_dim)
                self.autoencoder = autoencoder(code_dim=self.code_dim)
            elif self.data_type == 'poisson_digit':
                autoencoder_model_path = './checkpoint_autoencoder/autoencoder_poisson_dim{}.pth'.format(self.code_dim)
                self.autoencoder = autoencoder(input_dim=self.img_size**2 * 3, code_dim=self.code_dim)
            elif self.data_type == 'cifar10':
                autoencoder_model_path = './checkpoint_autoencoder/autoencoder_cifar10.pkl'
                self.autoencoder = autoencoder_cifar10()
            self.autoencoder.eval()
            self.autoencoder.to(self.device)


            autoencoder_checkpoint = torch.load(autoencoder_model_path)
            self.autoencoder.load_state_dict(autoencoder_checkpoint)

            for param in self.autoencoder.parameters():
                param.requires_grad = False 
        
        if self.data_type == 'gaussian2dgrid':
            self.generator = model_gaussian2d.GeneratorNet()
        elif self.data_type == 'mnist' or self.data_type == 'poisson_digit':
            if self.model_type == 'VGAN':
                self.generator = model_mnist_VGAN.GeneratorNet()
            elif self.model_type == 'DCGAN':
                self.generator = model_DCGAN.generator_mnist(z_size=self.z_size)
            else:
                self.generator = model_GAN_in_codespace.generator_mnist_in_code(z_size=self.z_size, code_dim=self.code_dim)
        elif self.data_type == 'svhn':
            if self.model_type == 'DCGAN':
                self.generator = model_DCGAN.generator_svhn(z_size=self.z_size)
        elif self.data_type == 'celeba':
            if self.model_type == 'DCGAN':
                self.generator = model_DCGAN.generator_celeba(z_size=self.z_size)
                self.generator.apply(model_DCGAN.weights_init)
            #elif self.model_type == 'GAN with AE':
            #    self.generator = model_GAN_in_codespace.generator_cifar10_in_code(z_size=self.z_size, code_dim=self.code_dim)
        elif self.data_type == 'cifar10':
            if self.model_type == 'DCGAN':
                self.generator = model_DCGAN.generator_cifar10(z_size=self.z_size)
            elif self.model_type == 'GAN with AE':
                self.generator = model_GAN_in_codespace.generator_cifar10_in_code(z_size=self.z_size, code_dim=self.code_dim)
            
        if self.model_type == 'DCGAN':
            if self.data_type == 'cifar10':
                self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
            else:
                self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        else:
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
            
        if self.data_type == 'cifar10':
            self.scheduler_g = optim.lr_scheduler.LambdaLR(self.g_optimizer, lambda step: 1 - step / (len(self.data_loader) * self.num_epochs))
        else:
            self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=self.lr_gamma)
        self.generator.to(self.device)
        
        self.clf = KernelClassifier(dim=self.dim, kernel=self.kernel, gamma=self.gamma, gamma_ratio=self.gamma_ratio, alpha=self.alpha, lmbda=self.lmbda, lmbda_IKL=self.lmbda_IKL, var_IKL=self.var_IKL, eta=self.eta, budget=self.budget, lr=self.lr, lr_IKL=self.lr_IKL, img_size=self.img_size, use_gpu=self.use_gpu, lossfn=self.lossfn, data_type=self.data_type, model_type=self.model_type)
        
        
    def images_to_vectors(self, images):
        return images.view(images.size(0), -1)

    
    def vectors_to_images(self, vectors):
        if self.data_type == 'poisson_digit':
            return vectors.view(vectors.size(0), 1, self.img_size, self.img_size * 3)
        else:
            return vectors.view(vectors.size(0), 1, self.img_size, self.img_size)
    
    
    def noise(self, size):
        if (self.data_type == 'mnist' or self.data_type == 'celeba' or self.data_type == 'cifar10') and self.model_type == 'DCGAN':
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
        
        optimizer.zero_grad()
       
        prediction = self.discriminator(fake_data)
        prediction = prediction.reshape(-1, 1)
        prediction = prediction.to(self.device)
            
        if self.lossfn == 'logistic':
            error = - nn.LogSigmoid()(prediction).mean()
        elif self.lossfn == 'hinge':
            error = nn.ReLU()(1.0 - prediction).mean() #hinge loss
            #error = -prediction.mean()
        error.backward()
        
        optimizer.step()
        
        return error

    
    def evaluate_gaussian2d(self, epoch):
        
        if self.data.name == 'gmm2d':
            X, Y = np.meshgrid(np.linspace(-35, 35, 50), np.linspace(-20, 20, 100))
        else:
            X, Y = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        if self.use_gpu and torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            Y_tensor = Y_tensor.cuda()
        Z = self.discriminator_for_graph(X_tensor, Y_tensor)
        Z = Z.cpu().detach().numpy()
        real_t_data = self.data.data[:2500]
        fake_t_data = self.generator(self.noise(500))
        fake_t_data = fake_t_data.cpu().detach().numpy()
        fig, ax = plt.subplots()
        #CS = ax.contour(X, Y, Z)
        #ax.clabel(CS, inline=1, fontsize=10)
        fig.suptitle('2d gaussiangrid at epoch {}'.format(epoch))
        ax.scatter(real_t_data[:,0], real_t_data[:,1], marker='o', s=3)
        plt.axis('equal')
        ax.scatter(fake_t_data[:,0], fake_t_data[:,1], marker='o', s=3)
        plt.axis('equal')
        if self.data.name == 'gaussian_grid':
            ax.set_xlim([-6, 6])
            ax.set_ylim([-6, 6])
        elif self.data.name == 'swissroll' or self.data.name == '2spirals':
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
        elif self.data.name == 'gmm2d':
            pass
            #ax.set_xlim([-10, 10])
            #ax.set_ylim([-7, 7])
        else:
            ax.set_xlim([-2*self.data.r, 2*self.data.r])
            ax.set_ylim([-2*self.data.r, 2*self.data.r])

        if not os.path.exists('out_image/'):
            os.makedirs('out_image/')
        if not os.path.exists('out_image/out_gaussian2d_online_kernel'):
            os.makedirs('out_image/out_gaussian2d_online_kernel')

        plt.savefig('out_image/out_gaussian2d_online_kernel/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        if self.data.name is not 'swissroll' and self.data.name is not '2spirals':
            samples = self.generator(self.noise(self.num_samples)).cpu().detach().numpy()
            if self.data.name == 'circle2d':
                num_high_qual_samples, center_captured, reverse_kl = self.data.mode_collapse_metric(samples)
                print('The number of high quality samples among 2500 samples is', num_high_qual_samples)
                print('Is the center captured?', center_captured)
                print('The reverse kl divergence is', reverse_kl)
            elif self.data.name == 'gmm2d':
                reverse_kl = self.data.mode_collapse_metric(samples)
                print('The reverse kl divergence is', reverse_kl)
            else:
                num_modes, num_high_qual_samples, mode_counter, reverse_kl = self.data.mode_collapse_metric(samples)
                print('Total number of generated modes is ', num_modes)
                print('The number of high quality samples among 2500 samples is', num_high_qual_samples)
                print('The mode dictionary is', mode_counter)
                print('The reverse kl divergence is', reverse_kl)
            self.epoch_arr.append(epoch)
            self.reverse_kl.append(reverse_kl)
            if epoch == self.num_epochs:
                fig = plt.figure()
                ax = plt.axes()
                ax.plot(self.epoch_arr, self.reverse_kl)
                plt.title("Reverse KL Divergence Graph")
                plt.xlabel("epoch")
                plt.ylabel("reverse KL")
                plt.savefig('out_image/out_gaussian2d_online_kernel/reverse_KL_graph.png', bbox_inches='tight')
                plt.close(fig)

                f = open('result.txt', 'w')
                if self.data.name is not 'gmm2d':
                    if self.data.name == 'circle2d':
                        f.write('Is the center captured? ' + str(center_captured) + '\n')
                    else:
                        f.write('Total number of generated modes is ' + str(num_modes) + '\n')
                    f.write('The number of high quality samples among 2500 samples is ' + str(num_high_qual_samples) + '\n')
                f.write('The reverse kl divergence is ' + str(reverse_kl))
                f.close()

        
    def evaluate_image(self, epoch, n_batch):

        if self.data_type == 'poisson_digit':
            samples = self.generator(self.noise(16))
            fig = plt.figure(figsize=(16, 16))
            gs = gridspec.GridSpec(4, 4)
        else:
            samples = self.generator(self.noise(64))
            fig = plt.figure(figsize=(8, 8))
            gs = gridspec.GridSpec(8, 8)
            
        if self.model_type == 'GAN with AE':
            samples = self.autoencoder.decoder(samples)
        samples = samples.cpu().data.numpy()
        
        if self.data_type == 'mnist':
            samples = samples.reshape(64, self.img_size, self.img_size)
        elif self.data_type == 'poisson_digit':
            samples = samples.reshape(16, self.img_size, self.img_size * 3)
        elif self.data_type == 'svhn' or self.data_type == 'celeba' or self.data_type == 'cifar10':
            samples = samples.reshape(64, 3, self.img_size, self.img_size)
    
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if self.data_type == 'mnist' or self.data_type == 'poisson_digit':
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

            plt.savefig('out_image/out_mnist_online_kernel/epoch{}_nbatch{}.png'.format(str(epoch).zfill(3), str(n_batch).zfill(3)), bbox_inches='tight')
        elif self.data_type == 'poisson_digit':
            if not os.path.exists('out_image/out_poisson_digit_online_kernel'):
                os.makedirs('out_image/out_poisson_digit_online_kernel')

            plt.savefig('out_image/out_poisson_digit_online_kernel/epoch{}_nbatch{}.png'.format(str(epoch).zfill(3), str(n_batch).zfill(3)), bbox_inches='tight')
        elif self.data_type == 'svhn':
            if not os.path.exists('out_image/out_svhn_online_kernel'):
                os.makedirs('out_image/out_svhn_online_kernel')

            plt.savefig('out_image/out_svhn_online_kernel/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        elif self.data_type == 'celeba':
            if not os.path.exists('out_image/out_celeba_online_kernel'):
                os.makedirs('out_image/out_celeba_online_kernel')

            plt.savefig('out_image/out_celeba_online_kernel/epoch{}_nbatch{}.png'.format(str(epoch).zfill(3), str(n_batch).zfill(4)), bbox_inches='tight')
        elif self.data_type == 'cifar10':
            if not os.path.exists('out_image/out_cifar10_online_kernel'):
                os.makedirs('out_image/out_cifar10_online_kernel')

            plt.savefig('out_image/out_cifar10_online_kernel/epoch{}_nbatch{}.png'.format(str(epoch).zfill(3), str(n_batch).zfill(4)), bbox_inches='tight')
        plt.close(fig)
        
    
    def show_cyclicity(self, plotter, epoch):
        from sklearn.decomposition import PCA
        discriminator_reprs = np.array(self.discriminator_reprs)
        pca = PCA(n_components = 2)
        pca.fit(discriminator_reprs)
        ldr = pca.transform(discriminator_reprs)        
        n = ldr.shape[0]
        cmap = plotter.get_cmap("Reds")
        clear_output(wait=True)
        for i in range(n):
            plotter.scatter(ldr[i,0],ldr[i,1],color=cmap(i/n))
            if i < n - 1:
                plotter.plot(ldr[i:i+2,0], ldr[i:i+2,1], color=cmap((i+0.5)/n))   
        plotter.show()
        if not os.path.exists('out_image/'):
            os.makedirs('out_image/')
        if not os.path.exists('out_image/out_gaussian2d_online_kernel'):
            os.makedirs('out_image/out_gaussian2d_online_kernel')
        if not os.path.exists('out_image/out_gaussian2d_online_kernel/cycling_behavior'):
            os.makedirs('out_image/out_gaussian2d_online_kernel/cycling_behavior')
        plotter.savefig('out_image/out_gaussian2d_online_kernel/cycling_behavior/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    
    
    
    def calculate_reverse_KL(self, lenet_path, model_path=None): #calculate reverse KL divergence for PoissonDigitDataset
    
        if model_path != None:
            self.generator.eval()
            generator_checkpoint = torch.load(model_path)
            self.generator.load_state_dict(generator_checkpoint)
            
        le_classifier = LeNet5()
        le_classifier.eval()
        le_classifier.to(self.device)
        lenet_checkpoint = torch.load(lenet_path)
        le_classifier.load_state_dict(lenet_checkpoint)
        
        samples = self.generator(self.noise(10000))
        if self.model_type == 'GAN with AE':
            samples = self.autoencoder.decoder(samples)
        samples = samples.reshape(10000, self.img_size, self.img_size * 3)
        
        reverse_kl = self.data.reverse_kl_div(samples, le_classifier)
        print('Reverse KL divergence is', reverse_kl)
        
    
    
    def calculate_score(self, model_path=None):
        
        if model_path != None:
            self.generator.eval()
            generator_checkpoint = torch.load(model_path)
            self.generator.load_state_dict(generator_checkpoint)
        
        print('Start generating fake images.....')
        start = time.time()
        imgs = generate_imgs(self.generator, self.device, self.z_size, 50000, self.batch_size)
        print('Spent time for generating images is ', time.time() - start)
        print('Start calculating IS & FID.....')
        start = time.time()
        is_score, fid_score = get_inception_and_fid_score(imgs, self.device, './stats/cifar10_stats.npz', verbose=True)
        print('Inception score: ', is_score[0])
        print('Inception score std: ', is_score[1])
        print('FID score: ', fid_score)
        print('Spent time for calculating scores is ', time.time() - start)
        
            
        
    
    def train_GAN(self,plotter=False):  
        
        
        if plotter:
            totsteps = 0

            D_POINT_REPR_SIZE = 500 #256 #16384

            # Take a sample of training points from training set
            temp_data_loader = torch.utils.data.DataLoader(self.data, batch_size=D_POINT_REPR_SIZE, shuffle=True)

            randsample = iter(temp_data_loader).next().type(torch.FloatTensor)
            if self.use_gpu and torch.cuda.is_available(): 
                randsample = randsample.cuda()
            D_POINT_REPR_PTS = randsample   

            REPR_PTS = D_POINT_REPR_PTS        
            
        for epoch in range(1, 1 + self.num_epochs):
            epoch_start = time.time()

            for n_batch, (real_batch) in enumerate(tqdm(self.data_loader)):
                if self.data_type is not 'gaussian2dgrid' and self.data_type is not 'poisson_digit':
                    real_batch = real_batch[0] # real_batch is tuple for mnist, svhn, ... (data, target)
                if plotter:
                    totsteps += self.batch_size

                    if totsteps > 200:
                        # APPEND NEW FUNC VALS TO REPR pts 
                        self.discriminator_reprs.append(self.discriminator(REPR_PTS).detach().cpu().numpy())
                        totsteps = 0
                
                
                if self.data_type == 'gaussian2dgrid':
                    real_data = real_batch.type(torch.FloatTensor)
                elif self.data_type == 'poisson_digit':
                    real_batch = real_batch.type(torch.FloatTensor)
                    real_data = self.images_to_vectors(real_batch)
                elif self.data_type == 'mnist' and self.model_type is not 'DCGAN':
                    real_data = self.images_to_vectors(real_batch)
                else:
                    real_data = real_batch
                    
                real_data = real_data.to(self.device)
                    
                fake_data = self.generator(self.noise(real_data.size(0))).detach()
                
                # Train the encoder when we use DCGAN architecture to OKGAN
                if (self.model_type == 'DCGAN') and (epoch != 0 or n_batch != 0):
                    for i in range(self.e_steps):
                        fake_data = self.generator(self.noise(real_data.size(0))).detach()
                        e_error = self.clf.train_encoder(real_data, fake_data)
                
                # Train the generative model for the kernel when we use implicit kernel learning
                if (self.kernel == 'IKL') and (epoch != 0 or n_batch != 0):
                    for i in range(self.IKL_steps):
                        fake_data = self.generator(self.noise(real_data.size(0))).detach()
                        if self.model_type == 'GAN with AE':
                            IKL_error = self.clf.train_IKLNet(self.autoencoder.encoder(real_data), fake_data)
                        else:
                            IKL_error = self.clf.train_IKLNet(real_data, fake_data)
                        
                # Train the online kernel classifier
                if self.model_type == 'GAN with AE':
                    self.online_kernel(self.autoencoder.encoder(real_data), fake_data)
                else:
                    self.online_kernel(real_data, fake_data)
                
                # Train the generator
                for i in range(self.g_steps):
                    fake_data = self.generator(self.noise(real_data.size(0)))
                    g_error = self.train_generator(self.g_optimizer, fake_data)
                
                if self.data_type == 'celeba' or (self.data_type == 'mnist' and self.model_type == 'DCGAN') or self.data_type == 'cifar10':
                    if n_batch % 100 == 0 and n_batch != 0:
                        self.evaluate_image(epoch, n_batch)
                        if self.kernel == 'IKL':
                            if self.model_type == 'DCGAN':
                                print('[{}/{}][{}/{}]'.format(epoch, self.num_epochs, n_batch, len(self.data_loader)), 'gen loss :', g_error.item(), 'encoder loss :', e_error.item(), 'IKLNet loss :', IKL_error.item())
                            else:
                                print('[{}/{}][{}/{}]'.format(epoch, self.num_epochs, n_batch, len(self.data_loader)), 'gen loss :', g_error.item(), 'IKLNet loss :', IKL_error.item())
                        else:
                            print('[{}/{}][{}/{}]'.format(epoch, self.num_epochs, n_batch, len(self.data_loader)), 'gen loss :', g_error.item(), 'encoder loss :', e_error.item())


            if epoch > 2 and plotter:
                self.show_cyclicity(plotter, epoch)
                            
            #self.scheduler_g.step()
            if not os.path.exists('checkpoint/'):
                os.makedirs('checkpoint/')
            
            if self.data_type == 'gaussian2dgrid':
                checkpoint_folder = 'checkpoint/checkpoint_gaussian2dgrid_online_kernel'
            elif self.data_type == 'mnist':
                checkpoint_folder = 'checkpoint/checkpoint_mnist_online_kernel'
            elif self.data_type == 'poisson_digit':
                checkpoint_folder = 'checkpoint/checkpoint_poisson_digit_online_kernel'
            elif self.data_type == 'svhn':
                checkpoint_folder = 'checkpoint/checkpoint_svhn'
            elif self.data_type == 'celeba':
                checkpoint_folder = 'checkpoint/checkpoint_celeba'
            elif self.data_type == 'cifar10':
                checkpoint_folder = 'checkpoint/checkpoint_cifar10'
                
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            torch.save(self.generator.state_dict(), os.path.join(checkpoint_folder, 'gen_{}'.format(str(epoch).zfill(3))))
            if self.data_type == 'cifar10' and self.model_type == 'DCGAN':
                self.clf.save_encoder(epoch)
                self.clf.save_online_kernel()
                
            if epoch % self.division == 0:
                print('spent time for epoch {} is {}s'.format(epoch, time.time()-epoch_start))
                if self.model_type == 'DCGAN':
                    if self.kernel == 'IKL':
                        print('epoch # :', epoch, 'gen loss :', g_error.item(), 'encoder loss :', e_error.item(), 'IKLNet loss :', IKL_error.item())
                    else:
                        print('epoch # :', epoch, 'gen loss :', g_error.item(), 'encoder loss :', e_error.item())
                else:
                    if self.kernel == 'IKL':
                        print('epoch # :', epoch, 'gen loss :', g_error.item(), 'IKLNet loss :', IKL_error.item())
                    else:
                        print('epoch # :', epoch, 'gamma :', self.clf.gamma, 'gen loss :', g_error.item())
            
            
                if self.data_type == 'gaussian2dgrid':
                    self.evaluate_gaussian2d(epoch)
                else:
                    self.evaluate_image(epoch, n_batch)
                
            if self.data_type == 'gaussian2dgrid':
                self.clf.gamma_update()
                
                if self.lr_schedule != None:
                    for (lr_epoch, new_lr) in self.lr_schedule:
                        if epoch == lr_epoch:
                            print(f'LR to {new_lr}')
                            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=new_lr)
                
                if self.lr_IKL_schedule != None:
                    for (lr_epoch, new_lr) in self.lr_IKL_schedule:
                        if epoch == lr_epoch:
                            print(f'IKL_LR to {new_lr}')
                            self.clf.lr_IKL_update(new_lr=new_lr)
