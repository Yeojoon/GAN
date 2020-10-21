import torch
import torch.nn as nn
import torch.nn.functional as F


'''
class generator_mnist_in_code(nn.Module):
    def __init__(self, z_size, code_dim):
        super(generator_mnist_in_code, self).__init__()
        self.fc1 = nn.Linear(z_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, code_dim)
        #self.fc5 = nn.Linear(784, code_dim)

    def forward(self, samples):
        x = nn.LeakyReLU(0.2)(self.fc1(samples))
        x = nn.LeakyReLU(0.2)(self.fc2(x))
        x = nn.LeakyReLU(0.2)(self.fc3(x))
        x = nn.Tanh()(self.fc4(x))
        #x = nn.Tanh()(self.fc5(x))
        return x
    

    
class discriminator_mnist_in_code(nn.Module):
    def __init__(self, code_dim):
        super(discriminator_mnist_in_code, self).__init__()
        n_out = 1
        self.fc1 = nn.Linear(code_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, n_out)
        #self.fc5 = nn.Linear(784, code_dim)

    def forward(self, samples):
        x = nn.Dropout(0.3)(nn.LeakyReLU(0.2)(self.fc1(samples)))
        x = nn.Dropout(0.3)(nn.LeakyReLU(0.2)(self.fc2(x)))
        x = nn.Dropout(0.3)(nn.LeakyReLU(0.2)(self.fc3(x)))
        x = nn.Sigmoid()(self.fc4(x))
        #x = nn.Tanh()(self.fc5(x))
        return x
'''



class generator_mnist_in_code(nn.Module):
    def __init__(self, z_size, code_dim):
        super(generator_mnist_in_code, self).__init__()
        self.fc1 = nn.Linear(z_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, code_dim)
        #self.fc5 = nn.Linear(784, code_dim)

    def forward(self, samples):
        x = nn.ReLU()(self.fc1(samples))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = nn.Tanh()(self.fc4(x))
        #x = nn.Tanh()(self.fc5(x))
        return x
    

    
class discriminator_mnist_in_code(nn.Module):
    def __init__(self, code_dim):
        super(discriminator_mnist_in_code, self).__init__()
        n_out = 1
        self.fc1 = nn.Linear(code_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, n_out)
        #self.fc5 = nn.Linear(784, code_dim)

    def forward(self, samples):
        x = nn.ReLU()(self.fc1(samples))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = nn.Sigmoid()(self.fc4(x))
        #x = nn.Tanh()(self.fc5(x))
        return x