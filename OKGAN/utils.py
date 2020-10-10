import random

import torch
import numpy as np


def generate_imgs(net_G, device, z_dim=128, size=5000, batch_size=128):
    net_G.eval()
    imgs = []
    with torch.no_grad():
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            z = torch.randn(end - start, z_dim, 1, 1).to(device)
            imgs.append(net_G(z).cpu().numpy())
    net_G.train()
    imgs = np.concatenate(imgs, axis=0)
    imgs = (imgs + 1) / 2
    return imgs