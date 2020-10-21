import torch
from torch import nn, optim



class IKLNet(torch.nn.Module):
    """
    A generative model for the spectral distribution of the kernel
    """
    def __init__(self, n_in=16, n_out=16):
        super(IKLNet, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(32, n_out),
        #    nn.ReLU()
        )
        #self.out = nn.Sequential(
        #    nn.Linear(32, n_out),
            #nn.Tanh()
        #)
        

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        #x = self.out(x)
        return x
    
    
    
class IKLNet_cifar10(torch.nn.Module):
    """
    A generative model for the spectral distribution of the kernel
    """
    def __init__(self, n_in=16, n_out=16):
        super(IKLNet_cifar10, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(32, n_out),
            #nn.Tanh()
        )
        

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)
        return x