import torch
from torch import nn, optim
import model.model_DCGAN as model_DCGAN

batch_size_train = 128
batch_size_test = 1000
log_interval = 10

class KernelClassifier:
    def __init__(self,dim,kernel='gaussian',gamma=1.0, gamma_ratio=1.002, budget=512,
                 lmbda=1.0,eta=0.01, margin=1.0, degree=3, coef0=0.0, alpha=0.5, lr=5e-4, img_size=32, use_gpu=True, lossfn = 'logistic', data_type='gaussian2dgrid'):
        self.lossfn = lossfn
        self.kern=kernel
        self.budget = budget
        self.lmbda = lmbda
        self.eta = eta
        self.margin = margin
        self.dim = dim
        self.gamma = gamma
        self.gamma_ratio = gamma_ratio
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.lr = lr
        self.img_size = img_size
        self.use_gpu = use_gpu
        self.data_type = data_type
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.alphas = torch.zeros(self.budget,device=self.device)
        if self.data_type == 'gaussian2dgrid' or self.data_type == 'mnist':
            self.keypoints = torch.zeros(self.budget,self.dim, device=self.device)
        else:
            self.keypoints = torch.zeros((self.budget, 3, self.img_size, self.img_size), device=self.device)
        self.pointer = 0
        self.offset = 0.0
        
        if self.data_type == 'gaussian2dgrid':
            self.z_size = 2
        else:
            self.z_size = 100
        
        if self.data_type == 'svhn':
            self.encoder = model_DCGAN.encoder_svhn(z_size=self.z_size)
            self.e_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.encoder.to(self.device)
    
    
    def gamma_update(self):
        
        self.gamma *= self.gamma_ratio
    
    
    def kernel(self,X):
        if self.kern == 'linear':
            return self.kernel_linear(X)
        elif self.kern == 'gaussian':
            return self.kernel_gaussian(X)
        elif self.kern == 'poly':
            return self.kernel_poly(X)
        elif self.kern == 'rq':
            return self.kernel_rq(X)
        else:
            raise Exception('No kernel specified')
      
    
    def kernel_gaussian(self,X):
        Z = self.keypoints
        if self.data_type == 'svhn':
            X = self.encoder(X)
            Z = self.encoder(Z)
        Xnorms = (X*X).sum(dim=1)
        Znorms = (Z*Z).sum(dim=1)
        onesn = torch.ones(X.shape[0],device=self.device)
        onesk = torch.ones(Z.shape[0],device=self.device)
        M = torch.ger(Xnorms,onesk) + torch.ger(onesn,Znorms) - 2 * (X @ Z.T)
        return torch.exp(-self.gamma*M)


    def kernel_linear(self,X):
        Z = self.keypoints
        if self.data_type == 'svhn':
            X = self.encoder(X)
            Z = self.encoder(Z)
        return X @ Z.T 
    
    
    def kernel_poly(self, X):
        Z = self.keypoints
        if self.data_type == 'svhn':
            X = self.encoder(X)
            Z = self.encoder(Z)
        return (self.gamma * X @ Z.T + self.coef0)**self.degree
    
    
    def kernel_rq(self, X): #rational quadratic
        Z = self.keypoints
        if self.data_type == 'svhn':
            X = self.encoder(X)
            Z = self.encoder(Z)
        Xnorms = (X*X).sum(dim=1)
        Znorms = (Z*Z).sum(dim=1)
        onesn = torch.ones(X.shape[0],device=self.device)
        onesk = torch.ones(Z.shape[0],device=self.device)
        M = torch.ger(Xnorms,onesk) + torch.ger(onesn,Znorms) - 2 * (X @ Z.T)
        return (1+M/(2*self.alpha))**(-self.alpha)
    
        
    def funceval(self,X):
        kernel_vals = self.kernel(X)
#         print(kernel_vals)
        if torch.isnan(kernel_vals).any():
            print("nan encountered")
            print(self.alphas)
            print(X)
            raise Exception("nan encountered in funceval calculation")
        return kernel_vals @ self.alphas + self.offset
    
    
    def predict_proba(self,X):
        vals = self.funceval(X)
        #expnegtwovals = torch.exp(-vals)
        #probs = 1/(1 + expnegtwovals)
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
    
    
    def train_encoder(self, real_data, fake_data):
        # Reset gradients
        self.e_optimizer.zero_grad()
        # Sample noise and generate fake data
        #start = time.time()
        prediction_real = self.funceval(real_data)
        prediction_fake = self.funceval(fake_data)
        prediction_real = prediction_real.reshape(-1, 1)
        prediction_fake = prediction_fake.reshape(-1, 1)
        if self.use_gpu and torch.cuda.is_available():
            prediction_real = prediction_real.cuda()
            prediction_fake = prediction_fake.cuda()
        #print('spent time for getting D when training encoder is {}'.format(time.time()-start))
        # Calculate error and backpropagate
        error = (nn.ReLU()(1.0 - prediction_real) + nn.ReLU()(1.0 + prediction_fake)).mean() #hinge loss

        #start = time.time()
        error.backward()
        #print('spent time for backpropagation when training encoder is {}'.format(time.time()-start))
        # Update weights with gradients
        self.e_optimizer.step()
        # Return error
        return error
    
    
    def train(self,train_loader,test_loader=False,num_epochs=5):
        train_losses = []
        train_counter = []
        if test_loader: self.test(test_loader)
        for epoch in range(1,num_epochs+1):           
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device).long()
                #print(data.shape)
                #print(target.shape)
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
            data, target = data.to(self.device), target.to(self.device).long()
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
            #lss = torch.log(1 + torch.exp(-Y*vals))
            lss = -torch.log(torch.sigmoid(-Y*vals))
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
            
            newalphas = self.eta * Y * torch.sigmoid(-funcvals*Y)
            #newalphas = self.eta * Y * torch.exp(-funcvals*Y) / (1 + torch.exp(-funcvals*Y))
            #newalphas =   self.eta * ((Y + 1)/2 - self.predict_proba(X))
            
            self.offset += self.eta * (Y * torch.sigmoid(-funcvals*Y)).mean()
            #self.offset += self.eta * (Y * torch.exp(-funcvals*Y) / (1 + torch.exp(-funcvals*Y))).mean()
            #self.offset += self.eta * ((Y + 1)/2 - self.predict_proba(X)).mean()
            if self.data_type == 'svhn':
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
            """
            loss(theta,x,y) = max(0,1-yf_theta(x))
            loss'(theta,x,y) = 1[yf_theta(x) < 1](yf_theta(x))\nabla_theta f_theta)            
            """
            sigma = torch.where(Y*funcvals<=self.margin, torch.ones(1, device=self.device), torch.zeros(1, device=self.device))
            newalphas = self.eta * sigma * Y  
            
            self.offset += self.eta * (sigma * Y).mean()
            
            if self.data_type == 'svhn':
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
            #self.keypoints[a_start:a_end,:] = newkeypoints[b_start:b_end,:]
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
