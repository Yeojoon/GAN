import torch

batch_size_train = 128
batch_size_test = 1000
log_interval = 10

class KernelClassifier:
    def __init__(self,dim,kernel='gaussian',gamma=1.0,budget=512,
                 lmbda=1.0,eta=0.01, device=torch.device("cuda"), lossfn = 'hinge'):
        self.lossfn = lossfn
        self.device = device
        self.kern=kernel
        self.budget = budget
        self.lmbda = lmbda
        self.eta = eta
        self.dim = dim
        self.gamma = gamma
        self.alphas = torch.zeros(self.budget,device=self.device)
        self.keypoints = torch.zeros(self.budget,self.dim,device=self.device)
        self.pointer = 0
        self.offset = 0.0
    
    def kernel(self,X):
        if self.kern == 'linear':
            return self.kernel_linear(X)
        elif self.kern == 'gaussian':
            return self.kernel_gaussian(X)
        else:
            raise Exception('No kernel specified')
        
    def kernel_gaussian(self,X):
        Z = self.keypoints
        Xnorms = (X*X).sum(dim=1)
        Znorms = (Z*Z).sum(dim=1)
        onesn = torch.ones(X.shape[0],device=self.device)
        onesk = torch.ones(Z.shape[0],device=self.device)
        M = torch.ger(Xnorms,onesk) + torch.ger(onesn,Znorms) - 2 * (X @ Z.T)
        return torch.exp(-self.gamma*M)


    def kernel_linear(self,X):
        Z = self.keypoints
        return X @ Z.T 
        
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
        expnegtwovals = torch.exp(-vals)
        probs = 1/(1 + expnegtwovals)
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

    def train(self,train_loader,test_loader=False,num_epochs=5):
        train_losses = []
        train_counter = []
        if test_loader: self.test(test_loader)
        for epoch in range(1,num_epochs+1):           
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device).long()
                self.update(data,target)
                if batch_idx % log_interval == 0:
                    lossval = self.avgloss(data,target)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), lossval))
                    train_losses.append(lossval)
                    train_counter.append(
                        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
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
        lss = torch.log(1 + torch.exp(-Y*vals))
        if torch.isinf(lss).any():
            print(lss)
            print(-2*Y*vals)
            raise Exception("inf encountered in calculating losses")
        return lss
    
    def avgloss(self,X,Y):
        return torch.mean(self.losses(X,Y))
    
    def update(self,X,Y,verbose=False):
        
        if self.lossfn = 'logistic':

            newalphas =   self.eta * ((Y + 1)/2 - self.predict_proba(X))
            newkeypoints = X

            self.offset += self.eta * ((Y + 1)/2 - self.predict_proba(X)).mean()

            if verbose:
                print(self.predict_proba(X))
                print(Y)
                print(newalphas)

            if torch.isnan(newalphas).any():
                print("some alphas are nan")
                raise Exception("nan encountered in generating newalphas")
        elif self.lossfn = 'hinge':
            """
            loss(theta,x,y) = max(0,1-yf_theta(x))
            loss'(theta,x,y) = 1[yf_theta(x) < 1](yf_theta(x))\nabla_theta f_theta)            
            """
            funvals = self.funceval(X)
            newalphas = -self.eta * Y * funvals * (funvals < 1).int()  
            newkeypoints = X
        
        
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
            self.keypoints[a_start:a_end,:] = newkeypoints[b_start:b_end,:]
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
