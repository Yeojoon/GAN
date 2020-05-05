import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import argparse

class GeneratorNet(torch.nn.Module):
    
    
    def __init__(self, input_size=2, hidden_size=400, output_size=2):
        super(GeneratorNet, self).__init__()
        
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
            nn.BatchNorm1d(output_size),
            #nn.ReLU()
        )
        self.name = "GeneratorNet"

        
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        
        return x
