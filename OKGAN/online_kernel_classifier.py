import torch
from torch import nn, optim
import model.model_DCGAN as model_DCGAN
import model.model_IKL as model_IKL 

import os

batch_size_train = 128
batch_size_test = 1000
log_interval = 10

class KernelClassifier:
    def __init__(self,dim,kernel='gaussian',gamma=1.0, gamma_ratio=1.002, budget=512,
                 lmbda=1.0, lmbda_IKL=10, var_IKL=0.5**0.5, eta=0.01, margin=1.0, degree=3, coef0=0.0, alpha=0.5, lr=5e-4, lr_IKL=None, img_size=32, n_IKL_samples=128, use_gpu=True, lossfn = 'logistic', data_type='gaussian2dgrid', model_type='VGAN'):
        self.lossfn = lossfn
        self.kern=kernel
        self.budget = budget
        self.lmbda = lmbda
        self.lmbda_IKL = lmbda_IKL
        self.var_IKL = var_IKL         #variance constraint of IKL
        self.eta = eta
        self.margin = margin
        self.dim = dim
        self.use_gpu = use_gpu
        self.gamma = (gamma.cuda() if self.use_gpu and torch.cuda.is_available() and torch.is_tensor(gamma) else gamma) 
        self.gamma_ratio = gamma_ratio
        self.degree = degree
        self.coef0 = coef0
        self.alpha = (alpha.cuda() if self.use_gpu and torch.cuda.is_available() and torch.is_tensor(alpha) else alpha)
        self.lr = lr
        self.lr_IKL = lr_IKL
        self.img_size = img_size
        self.n_IKL_samples = n_IKL_samples
        self.data_type = data_type
        self.model_type = model_type
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.alphas = torch.zeros(self.budget,device=self.device)
        if self.model_type == 'VGAN' or self.model_type == 'GAN with AE':
            self.keypoints = torch.zeros(self.budget, self.dim, device=self.device)
        else:
            if self.data_type == 'mnist':
                if self.model_type == 'DCGAN':
                    self.keypoints = torch.zeros((self.budget, 1, self.img_size, self.img_size), device=self.device)
            else:
                self.keypoints = torch.zeros((self.budget, 3, self.img_size, self.img_size), device=self.device)
        self.pointer = 0
        self.offset = 0.0
        
        if self.data_type == 'gaussian2dgrid':
            self.z_size = 2
        else:
            if self.kern == 'IKL':
                self.z_size = self.dim
            else:
                self.z_size = 100
        
        if self.model_type == 'DCGAN':
            if self.data_type == 'mnist':
                self.encoder = model_DCGAN.encoder_mnist(z_size=self.z_size)
            elif self.data_type == 'svhn':
                self.encoder = model_DCGAN.encoder_svhn(z_size=self.z_size)
            elif self.data_type == 'celeba':
                self.encoder = model_DCGAN.encoder_celeba(z_size=self.z_size)
                self.encoder.apply(model_DCGAN.weights_init)
            elif self.data_type == 'cifar10':
                self.encoder = model_DCGAN.encoder_cifar10(z_size=self.z_size)
            
            if self.data_type == 'cifar10':
                self.e_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.9))
            else:
                self.e_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.encoder.to(self.device)
            
        if self.kern == 'IKL':
            if self.data_type == 'cifar10':
                self.IKLNet = model_IKL.IKLNet_cifar10(n_in=self.dim, n_out=self.dim)
            else:
                self.IKLNet = model_IKL.IKLNet(n_in=self.dim, n_out=self.dim)
            self.IKL_optimizer = optim.Adam(self.IKLNet.parameters(), lr=self.lr_IKL)
            self.IKLNet.to(self.device)
        
    
    
    def noise(self, size):
        n = torch.randn((size, self.dim), device=self.device)
        return n
    
    
    def gamma_update(self):
        
        self.gamma *= self.gamma_ratio
        
        
    def lr_IKL_update(self, new_lr):
        
        self.IKL_optimizer = optim.Adam(self.IKLNet.parameters(), lr=new_lr)
        
        
    def save_encoder(self, epoch):
        if not os.path.exists('checkpoint/'):
            os.makedirs('checkpoint/')
        if self.data_type == 'cifar10':
            checkpoint_folder = 'checkpoint/checkpoint_cifar10'
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint_folder, 'enc_{}'.format(str(epoch).zfill(3))))
    
    
    def save_online_kernel(self):
        if not os.path.exists('checkpoint/'):
            os.makedirs('checkpoint/')
        if self.data_type == 'cifar10':
            checkpoint_folder = 'checkpoint/checkpoint_cifar10'
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        torch.save(self.alphas, os.path.join(checkpoint_folder, 'alphas.pt'))
        torch.save(self.keypoints, os.path.join(checkpoint_folder, 'keypoints.pt'))
        torch.save(self.pointer, os.path.join(checkpoint_folder, 'pointer.pt'))
        torch.save(self.offset, os.path.join(checkpoint_folder, 'offset.pt'))
    
    
    def kernel(self,X):
        if self.kern == 'linear':
            return self.kernel_linear(X)
        elif self.kern == 'gaussian':
            return self.kernel_gaussian(X, self.gamma)
        elif self.kern == 'poly':
            return self.kernel_poly(X, self.gamma)
        elif self.kern == 'rq':
            return self.kernel_rq(X, self.alpha)
        elif self.kern == 'mixed_gaussian':
            return self.kernel_mixed_gaussian(X)
        elif self.kern == 'mixed_rq_linear':
            return self.kernel_mixed_rq_linear(X)
        elif self.kern == 'IKL':
            return self.kernel_IKL(X)
        else:
            raise Exception('No kernel specified')
      
    
    def kernel_gaussian(self,X, gamma):
        Z = self.keypoints
        if self.model_type == 'DCGAN':
            X = self.encoder(X)
            Z = self.encoder(Z)
        Xnorms = (X*X).sum(dim=1)
        Znorms = (Z*Z).sum(dim=1)
        onesn = torch.ones(X.shape[0],device=self.device)
        onesk = torch.ones(Z.shape[0],device=self.device)
        M = torch.ger(Xnorms,onesk) + torch.ger(onesn,Znorms) - 2 * (X @ torch.t(Z))
        if self.kern == 'mixed_gaussian':
            row, col = M.shape
            M = M.reshape(row, col, 1)
            return torch.exp(-gamma*M)
        return torch.exp(-gamma*M)


    def kernel_linear(self,X):
        Z = self.keypoints
        if self.model_type == 'DCGAN':
            X = self.encoder(X)
            Z = self.encoder(Z)
        return X @ torch.t(Z) 
    
    
    def kernel_poly(self, X, gamma):
        Z = self.keypoints
        if self.model_type == 'DCGAN':
            X = self.encoder(X)
            Z = self.encoder(Z)
        return (gamma * X @ torch.t(Z) + self.coef0)**self.degree
    
    
    def kernel_rq(self, X, alpha): #rational quadratic
        Z = self.keypoints
        if self.model_type == 'DCGAN':
            X = self.encoder(X)
            Z = self.encoder(Z)
        Xnorms = (X*X).sum(dim=1)
        Znorms = (Z*Z).sum(dim=1)
        onesn = torch.ones(X.shape[0],device=self.device)
        onesk = torch.ones(Z.shape[0],device=self.device)
        M = torch.ger(Xnorms,onesk) + torch.ger(onesn,Znorms) - 2 * (X @ torch.t(Z))
        if self.kern == 'mixed_rq_linear':
            row, col = M.shape
            M = M.reshape(row, col, 1)
            return (1+M/(2*alpha))**(-alpha)
        return (1+M/(2*alpha))**(-alpha)
    
    
    def kernel_mixed_gaussian(self, X):
        kernel_vals = self.kernel_gaussian(X, self.gamma)
        return kernel_vals.sum(dim=2)
    
    
    def kernel_mixed_rq_linear(self, X):
        kernel_vals = self.kernel_rq(X, self.alpha).sum(dim=2) + self.kernel_linear(X)
        return kernel_vals
    
    
    def kernel_IKL(self, X):
        Z = self.keypoints
        if self.model_type == 'DCGAN':
            X = self.encoder(X)
            Z = self.encoder(Z)
        out_IKL = self.IKLNet(self.noise(self.n_IKL_samples))
        phi_X = 1 / (self.n_IKL_samples)**0.5 * torch.cat((torch.cos(out_IKL @ torch.t(X)), torch.sin(out_IKL @ torch.t(X))), 0)
        phi_Z = 1 / (self.n_IKL_samples)**0.5 * torch.cat((torch.cos(out_IKL @ torch.t(Z)), torch.sin(out_IKL @ torch.t(Z))), 0)
        return torch.t(phi_X) @ phi_Z
            
    
    def funceval(self,X):
        kernel_vals = self.kernel(X)
        if torch.isnan(kernel_vals).any():
            print("nan encountered")
            print(self.alphas)
            print(X)
            raise Exception("nan encountered in funceval calculation")
        return kernel_vals @ self.alphas + self.offset
    
    
    def predict_proba(self,X):
        vals = self.funceval(X)
        probs = torch.sigmoid(-vals)
        if torch.isnan(probs).any():
            print(probs)
            print(vals)
            raise Exception("nan encountered in predict_proba calculation")
        return probs
    
    
    def predict(self,X):
        vals = self.funceval(X)
        return torch.sign(vals)
    
    
    def error(self,X,Y):
        signs = torch.sign(self.funceval(X) * Y)
        return torch.mean(1 - signs)/2
    
    
    def train_IKLNet(self, real_data, fake_data):
        
        self.IKL_optimizer.zero_grad()
        
        prediction_real = self.funceval(real_data)
        prediction_fake = self.funceval(fake_data)
        prediction_real = prediction_real.reshape(-1, 1).to(self.device)
        prediction_fake = prediction_fake.reshape(-1, 1).to(self.device)
        
        if self.lossfn == 'logistic':
            error = (nn.ReLU()(1.0 - prediction_real) + nn.ReLU()(1.0 + prediction_fake)).mean()
            #error = -(nn.LogSigmoid()(prediction_real).mean() + nn.LogSigmoid()(-prediction_fake).mean()) - self.lmbda_IKL * ((torch.norm(self.IKLNet(self.noise(self.n_IKL_samples)), dim=1)**2).mean() - self.var_IKL)**2
        elif self.lossfn == 'hinge':
            error = (nn.ReLU()(1.0 - prediction_real) + nn.ReLU()(1.0 + prediction_fake)).mean()
            #error = (nn.ReLU()(1.0 - prediction_real) + nn.ReLU()(1.0 + prediction_fake)).mean() - self.lmbda_IKL * ((torch.norm(self.IKLNet(self.noise(self.n_IKL_samples)), dim=1)**2).mean() - self.var_IKL)**2
            #error = (nn.ReLU()(1.0 - prediction_real) + nn.ReLU()(1.0 + prediction_fake)).mean() - self.lmbda_IKL * nn.ReLU()(self.var_IKL - torch.norm(self.IKLNet(self.noise(self.n_IKL_samples)), dim=1)).mean()
        error.backward()
        
        self.IKL_optimizer.step()
        
        return error
    
    
    def train_encoder(self, real_data, fake_data):
        
        self.e_optimizer.zero_grad()
        
        prediction_real = self.funceval(real_data)
        prediction_fake = self.funceval(fake_data)
        prediction_real = prediction_real.reshape(-1, 1).to(self.device)
        prediction_fake = prediction_fake.reshape(-1, 1).to(self.device)
        
        if self.lossfn == 'logistic':
            error = -(nn.LogSigmoid()(prediction_real).mean() + nn.LogSigmoid()(-prediction_fake).mean())
        elif self.lossfn == 'hinge':
            error = (nn.ReLU()(1.0 - prediction_real) + nn.ReLU()(1.0 + prediction_fake)).mean() #hinge loss
        error.backward()
        
        self.e_optimizer.step()
        
        return error
    
    
    def train(self,train_loader,test_loader=False,num_epochs=5):
        train_losses = []
        train_counter = []
        if test_loader: self.test(test_loader)
        for epoch in range(1,num_epochs+1):           
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.update(data,target)
                '''
                if batch_idx % log_interval == 0:
                    lossval = self.avgloss(data,target)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), lossval))
                    train_losses.append(lossval)
                    train_counter.append(
                        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                '''
            if test_loader: self.test(test_loader)

                
    def test(self,test_loader):
        test_losses, test_errors, test_loss, error_rate, counter = [], [], 0, 0, 0
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            test_loss += self.avgloss(data,target)
            error_rate += self.error(data,target)
            counter += 1
        test_loss /= counter
        error_rate /= counter
        test_losses.append(test_loss)
        test_errors.append(error_rate)
        print('\nTest set: Avg. loss: {:.4f}, Error: {:.4f}\n'.format(test_loss, error_rate))
    
    
    def losses(self,X,Y):
        vals = self.funceval(X)
        if self.lossfn == 'logistic':
            #lss = -torch.log(torch.sigmoid(Y*vals))
            lss = -nn.LogSigmoid()(Y*vals)
        elif self.lossfn == 'hinge':
            zeros = torch.zeros(len(Y), device=self.device)
            lss_vals = self.margin - Y*vals
            lss = torch.max(zeros, lss_vals)
        if torch.isinf(lss).any():
            print(lss)
            print(-2*Y*vals)
            raise Exception("inf encountered in calculating losses")
        return lss
    
    
    def avgloss(self,X,Y):
        return torch.mean(self.losses(X,Y))
    
    
    def update(self,X,Y,verbose=False):
        funcvals = self.funceval(X)
        self.alphas *= (1 - self.eta * self.lmbda)
        
        if self.lossfn == 'logistic':
            
            newalphas = self.eta * Y * nn.Sigmoid()(-funcvals*Y)
            self.offset += self.eta * (Y * nn.Sigmoid()(-funcvals*Y)).mean()
            
            if self.model_type == 'DCGAN' or self.kern == 'IKL':
                newalphas = newalphas.detach()
                self.offset = self.offset.detach()

            if verbose:
                print(self.predict_proba(X))
                print(Y)
                print(newalphas)

            if torch.isnan(newalphas).any():
                print("some alphas are nan")
                raise Exception("nan encountered in generating newalphas")
                
        elif self.lossfn == 'hinge':
            
            sigma = torch.where(Y*funcvals<=self.margin, torch.ones(1, device=self.device), torch.zeros(1, device=self.device))
            newalphas = self.eta * sigma * Y  
            
            self.offset += self.eta * (sigma * Y).mean()
            
            if self.model_type == 'DCGAN' or self.kern == 'IKL':
                newalphas = newalphas.detach()
                self.offset = self.offset.detach()
        
        
        self._insert_keypoints_alphas(X,newalphas)
    
    
    def _insert_keypoints_alphas(self,newkeypoints,newalphas):
    
        """
        This function aims to insert a new set of new "keypoints"
        as well as associated alphas. But in trying to do this efficiently,
        I'm keeping a single tensor of all the keypoints, with a pointer
        to the "oldest" example that was added as a keypoint. Recall that
        self.budget is the total number of keypoints we store.
        There will be k new
        keypoints inserted into the datastructure starting at self.pointer.
        But because of wrap-around issues, there might be need to start again at
        the beginning of the tensor, so I have to check if 
        """
    
        k = newkeypoints.shape[0]

        if self.pointer + k > self.budget:
            a1_start, a1_end = self.pointer, self.budget
            b1_start, b1_end = 0, self.budget - self.pointer
            a2_start, a2_end = 0, self.pointer + k - self.budget
            b2_start, b2_end = b1_end, k
            segments = [(a1_start, a1_end, b1_start, b1_end),
                        (a2_start, a2_end, b2_start, b2_end)]
            newpointer = a2_end 
        else:
            a_start, a_end = self.pointer, self.pointer + k
            b_start, b_end = 0, k
            segments = [(a_start, a_end, b_start, b_end)]
            newpointer = a_end if a_end < self.budget else 0
        for a_start, a_end, b_start, b_end in segments:
            self.keypoints[a_start:a_end] = newkeypoints[b_start:b_end]
            self.alphas[a_start:a_end] = newalphas[b_start:b_end]

        self.pointer = newpointer
                

def make_2d_dataset(num_pts=4096,batch_size=64,dimension=2):
    X1 = torch.randn(num_pts, dimension,device=torch.device("cuda"))
    X2 = torch.randn(num_pts, dimension, device=torch.device("cuda")) + 3.5
    X = torch.cat([X1, X2], dim=0)
    Y1 = -1*torch.ones(num_pts,device=torch.device("cuda"))
    Y2 = torch.ones(num_pts,device=torch.device("cuda"))
    Y = torch.cat([Y1, Y2], dim=0)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return loader
