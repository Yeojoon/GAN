import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from model_kernel_gaussian2d_from_PacGAN import *
from data.gaussianGridDataset import gaussianGridDataset
from data.ringDataset import ringDataset
from data.circleDataset import circleDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def noise(size):
    return Variable(torch.randn(size, 2))

nsize = 10000
alpha = 0.02
#plot = "grid"
fig, axs = plt.subplots(2, 4)

# ring2d, grid2d, circle2d

#off = 0
off = 1

## RING2D
g = GeneratorNet(final_bn=False)
g.load_state_dict(torch.load("figures/OKGAN_gen_2dring"))

#t = gaussianGridDataset(n=5, n_data=100, sig=0.01)
t = ringDataset(n=8, n_data=(nsize//8), r=4)
axs[0,0+off].scatter(t.data[:,0], t.data[:,1], s=50, alpha=alpha, linewidth=0)
axs[0,0+off].set_xlim([-6, 6])
axs[0,0+off].set_ylim([-6, 6])
axs[0,0+off].set_aspect("equal")
axs[0,0+off].get_xaxis().set_visible(False)
axs[0,0+off].get_yaxis().set_visible(False)

d = g(noise(nsize)).detach().numpy()
axs[1,0+off].scatter(d[:,0], d[:,1], s=50, alpha=alpha, linewidth=0)
axs[1,0+off].set_xlim([-6, 6])
axs[1,0+off].set_ylim([-6, 6])
axs[1,0+off].set_aspect("equal")
axs[1,0+off].get_xaxis().set_visible(False)
axs[1,0+off].get_yaxis().set_visible(False)


## GRID2D
g = GeneratorNet(final_bn=False)
g.load_state_dict(torch.load("figures/OKGAN_gen_2dgrid"))

t = gaussianGridDataset(n=5, n_data=(nsize//25), sig=0.01)
#t = ringDataset(n=8, n_data=200, r=4)
axs[0,1+off].scatter(t.data[:,0], t.data[:,1], s=50, alpha=alpha, linewidth=0)
axs[0,1+off].set_xlim([-6, 6])
axs[0,1+off].set_ylim([-6, 6])
axs[0,1+off].set_aspect("equal")
axs[0,1+off].get_xaxis().set_visible(False)
axs[0,1+off].get_yaxis().set_visible(False)

d = g(noise(nsize)).detach().numpy()
axs[1,1+off].scatter(d[:,0], d[:,1], s=50, alpha=alpha, linewidth=0)
axs[1,1+off].set_xlim([-6, 6])
axs[1,1+off].set_ylim([-6, 6])
axs[1,1+off].set_aspect("equal")
axs[1,1+off].get_xaxis().set_visible(False)
axs[1,1+off].get_yaxis().set_visible(False)


## CIRCLE2D
g = GeneratorNet(final_bn=False)
g.load_state_dict(torch.load("figures/OKGAN_gen_2dcircle"))

t = circleDataset(n_data=nsize, r=2)
#t = ringDataset(n=8, n_data=200, r=4)
axs[0,2+off].scatter(t.data[:,0], t.data[:,1], s=50, alpha=alpha, linewidth=0)
axs[0,2+off].set_xlim([-3, 3])
axs[0,2+off].set_ylim([-3, 3])
axs[0,2+off].set_aspect("equal")
axs[0,2+off].get_xaxis().set_visible(False)
axs[0,2+off].get_yaxis().set_visible(False)

d = g(noise(nsize)).detach().numpy()
axs[1,2+off].scatter(d[:,0], d[:,1], s=50, alpha=alpha, linewidth=0)
axs[1,2+off].set_xlim([-3, 3])
axs[1,2+off].set_ylim([-3, 3])
axs[1,2+off].set_aspect("equal")
axs[1,2+off].get_xaxis().set_visible(False)
axs[1,2+off].get_yaxis().set_visible(False)

axs[0,0+off].set_title("2D Ring", {"fontsize":25}, y=1.08)
axs[0,1+off].set_title("2D Grid", {"fontsize":25}, y=1.08)
axs[0,2+off].set_title("2D Circle", {"fontsize":25}, y=1.08)

axs[0,0].set_xlim([-1,1])
axs[0,0].set_ylim([-1,1])
axs[0,0].set_aspect("equal")
axs[0,0].axis("off")
axs[1,0].set_xlim([-1,1])
axs[1,0].set_ylim([-1,1])
axs[1,0].set_aspect("equal")
axs[1,0].axis("off")

axs[0,0].set_title("Target", {"fontsize":25}, y=0.42, x=0.7)
axs[1,0].set_title("OKGAN", {"fontsize":25}, y=0.42, x=0.7)

fig.set_size_inches(4*4, 4*2)
plt.show()

if False:
    for plot in ["grid","ring","circle"]:
        path = f"figures/OKGAN_gen_2d{plot}"
        
        g = GeneratorNet(final_bn=False)
        
        #if plot == "circle":
        #    path = "figures/gen_1999"
        #    g = GeneratorNet(final_bn=True)        

        g.load_state_dict(torch.load(path))

        

        v = g(noise(10000))

        d = v.detach().numpy()

        plt.scatter(d[:,0], d[:,1], s=50, alpha=alpha, linewidth=0)
        #plt.axis('equal')
        if plot == "circle":
            plt.xlim([-3, 3])
            plt.ylim([-3, 3])
            ##plt.xlim([-6, 6])
            ##plt.ylim([-6, 6])
        elif plot == "ring":
            plt.xlim([-6, 6])
            plt.ylim([-6, 6])
        elif plot == "grid":
            plt.xlim([-6, 6])
            plt.ylim([-6, 6])
        plt.axes().set_aspect("equal")
        plt.gcf().set_size_inches(4, 4, forward=True)
        plt.savefig(f"figures/2d_{plot}_figure.png")
        plt.close()
