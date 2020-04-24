import numpy as np
from torch.utils.data import Dataset, DataLoader

class ringDataset(Dataset):
    def __init__(self, n, n_data, sig = 0.01 ** 2, r = 1):
        self.data = None
        sig *= (r ** 2) # scaling
        for i in range(n):
            mean_x = r * np.cos(i * np.pi * 2 / n)
            mean_y = r * np.sin(i * np.pi * 2 / n)
            data = np.random.multivariate_normal((mean_x, mean_y), cov=(sig * np.eye(2)), size=n_data)
            if self.data is None:
                self.data = data
            else:
                self.data = np.concatenate((self.data, data), axis=0)
        self.out_dim = 2
        self.n_data = self.data.shape[0]
        self.name = "ring2d"
    
    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data
