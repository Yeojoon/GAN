import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import model_kernel_gaussian2d as model
import pyro
from pyro.contrib.gp.kernels.isotropic import RBF
from pyro.contrib.gp.kernels import DotProduct
from pyro.contrib.gp.kernels.dot_product import Polynomial
from sklearn.svm import SVC
from data.XORdataset import XORdataset_uniform, XORdataset_gaussian

from visualizer import plot_outputdata, checkviz
import time

import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
def poly_dotproduct(x, data):
    
    variance = 316**2
    bias = 0.0104**2
    d = 2
    
    return variance * torch.pow(bias + torch.dot(x, data), d)
'''

def exp(x, data):
    
    #k = 2.0814
    variance = 1.0
    l = 1.0
    
    return variance * torch.exp(-torch.norm(x - data)**2/(2 * l**2))


def kernel(data):
    
    pyro.clear_param_store()
    covar_module = RBF(input_dim=len(data[0]), variance=torch.tensor(316**2), lengthscale=torch.tensor(1.25))
    covar = covar_module(data, data)
    if torch.cuda.is_available():
        return covar.cuda()
    
    return covar


def XOR(a, b):
    
    threshold = 0
    if a >= threshold and b >= threshold:
        return int(0)
    elif a >= threshold and b < threshold:
        return int(1)
    elif a < threshold and b >= threshold:
        return int(1)
    else:
        return int(0)
    
    
def y_XOR(data):
    
    vec_XOR = np.vectorize(XOR)
    y = vec_XOR(data[:, 0], data[:, 1])
    
    return y
        
    
def get_alpha(data):
    
    data = data.cpu().detach().numpy()
    y = y_XOR(data)
    clf = SVC()
    clf.fit(data, y)
    alpha = clf.dual_coef_.reshape(-1)
    rho = clf.intercept_
    support_vecs = clf.support_vectors_
    
    return alpha, rho, support_vecs


def kernel_func(x, y, data, alpha, rho):
    
    N = len(data)
    
    variance = 1
    gamma = 1/len(data[0])
    z = 0
    
    for i in range(N):
        z += alpha[i] * variance * np.exp(-gamma * ((x-data[i, 0])**2 + (y-data[i, 1])**2))
    
    z += rho
    
    return z

        
if __name__ == '__main__':
    
    n_data = 200
    # Load data
    original_data = np.random.RandomState(0).randn(n_data, 2)
    print('generated data is')
    print(original_data)
    
    data = torch.from_numpy(original_data)
    data = data.type(torch.FloatTensor)
    if torch.cuda.is_available(): 
        data = data.cuda()
    alpha, rho, support_vecs = get_alpha(data)
    print('alpha is')
    print(alpha)
    
    data = data.cpu().detach().numpy()
    
    N = 50
    x = np.linspace(-3, 3, N)
    y = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(x, y)
    
    Z = kernel_func(X, Y, support_vecs, alpha, rho)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
    ax.scatter(original_data[:, 0], original_data[:, 1], cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
    cb = fig.colorbar(cs)  
    plt.axis([-3, 3, -3, 3])
    plt.title('kernel method on XOR')
    plt.show()
