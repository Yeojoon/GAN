import torch.nn as nn
import torch.nn.functional as F
import torch
'''
class DeepMLP_G(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP_G, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.name = "DeepMLP_G"

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.1)
        x = F.leaky_relu(self.map2(x), 0.1)
        return self.map3(x)

class DeepMLP_D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP_D, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.name = "DeepMLP_D"

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.1)
        x = F.leaky_relu(self.map2(x), 0.1)
        return torch.sigmoid(self.map3(x))
'''
class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m



class DeepMLP_G(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP_G, self).__init__()
        self.hidden0 = nn.Sequential( 
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential( 
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential( 
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.hidden3 = nn.Sequential( 
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.out = nn.Sequential( 
            nn.Linear(hidden_size, output_size),
            #nn.BatchNorm1d(output_size),
            #nn.ReLU()
        )
        self.name = "DeepMLP_G"

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x
'''
class DeepMLP_D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP_D, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        self.name = "DeepMLP_D"

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
'''
class DeepMLP_D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP_D, self).__init__()
        self.hidden0 = Maxout(input_size, hidden_size, 5)
        self.hidden1 = Maxout(hidden_size, hidden_size, 5)
        self.hidden2 = Maxout(hidden_size, hidden_size, 5)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        self.name = "DeepMLP_D"

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x