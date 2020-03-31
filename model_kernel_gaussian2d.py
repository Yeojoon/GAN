import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import argparse

class GeneratorNet(torch.nn.Module):
    
    
    def __init__(self, input_size=100, hidden_size=128, output_size=2):
        super(GeneratorNet, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
        self.name = "GeneratorNet"

        
    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.1)
        x = F.leaky_relu(self.map2(x), 0.1)
        
        return self.map3(x)
