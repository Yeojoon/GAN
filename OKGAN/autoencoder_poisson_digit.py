import os
import pandas as pd
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data.PoissonDigitDataset import PoissonDigitDataset

import model.model_autoencoder as model

import time

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='okgan')
parser.add_argument('--code_dim', type=int, default=16)

args = parser.parse_args()

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    if args.model == 'okgan':
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28*3)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 5e-4

img_size = 28

compose = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ])

dl = DataLoader(datasets.MNIST('./data/mnist', train=True, transform=compose, download=True), batch_size=60000, shuffle=False)
for _, full_batch in enumerate(dl):
    train_data = full_batch[0]
    train_labels = full_batch[1]

mnist_tensor = train_data.to(dtype=torch.float32)
mnist_tensor = mnist_tensor.reshape(mnist_tensor.shape[0], -1) 

mnist_df = pd.DataFrame(mnist_tensor.cpu().numpy())
mnist_df['label'] = train_labels.cpu().numpy()

MNIST_DIGITS = [
    mnist_df[
        mnist_df['label'] == ind
    ].drop('label',axis=1).values.reshape([-1,28,28]) 
    for ind in range(10)
]

def display_poisson_digits(mean=100,num_digits=3):
    # Here we pick a random integer from the poisson distribution
    digits = np.random.poisson(mean)
    # This is the list of the digits in this number
    digits_lst = list(map(int, list(str(digits))))
    num_dig = len(digits_lst)
    if num_digits < num_dig:
        return None
    digs = [0 for _ in range(num_digits - num_dig )] + digits_lst
    out = []
    # out is a list of mnist digits corres
    for dig in digs:
        digims = MNIST_DIGITS[dig]
        randind = np.random.randint(low=0,high=digims.shape[0])
        choice = digims[randind]
        out.append(choice)
    return np.hstack(out), digits


data_poisson_digit = PoissonDigitDataset(display_poisson_digits, n_data=100000) 
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data_poisson_digit, batch_size=batch_size, shuffle=True)

if args.model == 'okgan':
    model = model.autoencoder(input_dim=28*84, code_dim=args.code_dim)
elif args.model == 'gmmn':
    model = model.autoencoder_GMMN(code_dim=args.code_dim)
    
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in data_loader:
        img = data.type(torch.FloatTensor)
        img = img.view(img.size(0), -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        # ===================forward=====================
        if args.model == 'okgan':
            output = model(img)
        elif args.model == 'gmmn':
            _, output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_PoissonDigits{}.png'.format(epoch))

if not os.path.exists('checkpoint_autoencoder/'):
    os.makedirs('checkpoint_autoencoder/')
    
if args.model == 'okgan':
    torch.save(model.state_dict(), './checkpoint_autoencoder/autoencoder_poisson_dim{}.pth'.format(args.code_dim))
elif args.model == 'gmmn':
    torch.save(model.state_dict(), './checkpoint_autoencoder/autoencoder_gmmn_dim{}.pth'.format(args.code_dim))