import torch
import torch.nn as nn
import torch.nn.functional as F

class generator_mnist_in_code(nn.Module):
    def __init__(self, z_size, code_dim):
        super(generator_mnist_in_code, self).__init__()
        self.fc1 = nn.Linear(z_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, code_dim)
        #self.fc5 = nn.Linear(784, code_dim)

    def forward(self, samples):
        x = F.relu(self.fc1(samples))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = nn.Tanh()(self.fc4(x))
        #x = nn.Tanh()(self.fc5(x))
        return x
    

    
class generator_cifar10_in_code(nn.Module):
    def __init__(self, z_size, code_dim):
        super(generator_cifar10_in_code, self).__init__()
        self.fc1 = nn.Linear(z_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 784)
        self.fc5 = nn.Linear(784, code_dim)

    def forward(self, samples):
        x = F.relu(self.fc1(samples))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = nn.Sigmoid()(self.fc5(x))
        return x