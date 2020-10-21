import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import model.model_autoencoder as model

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
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

if args.model == 'okgan':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
elif args.model == 'gmmn':
    img_transform = transforms.Compose([transforms.ToTensor()])

dataset = MNIST('./mnist_data/dataset', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if args.model == 'okgan':
    model = model.autoencoder(code_dim=args.code_dim)
elif args.model == 'gmmn':
    model = model.autoencoder_GMMN(code_dim=args.code_dim)
    
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
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
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))

if not os.path.exists('checkpoint_autoencoder/'):
    os.makedirs('checkpoint_autoencoder/')
    
if args.model == 'okgan':
    torch.save(model.state_dict(), './checkpoint_autoencoder/autoencoder_dim{}.pth'.format(args.code_dim))
elif args.model == 'gmmn':
    torch.save(model.state_dict(), './checkpoint_autoencoder/autoencoder_gmmn_dim{}.pth'.format(args.code_dim))