import numpy as np
from torch.utils.data import Dataset, DataLoader

class circleDataset(Dataset):
    def __init__(self, n_data, sig = 0.06 ** 2, center_frac = 0.05, r = 1):
        angles = np.random.uniform(low=0, high=2*np.pi, size=n_data)
        sig *= (r ** 2) # scaling
        # zero out center_frac of the points
        zeros = 1.0 * (np.random.uniform(low=0, high=1, size=n_data) > 0.02)
        x = r * np.cos(angles) * zeros
        y = r * np.sin(angles) * zeros
        self.data = np.concatenate([x[:,None], y[:,None]], axis=1)
        self.data += np.random.multivariate_normal((0, 0), cov=(sig * np.eye(2)), size=n_data)
        self.out_dim = 2
        self.n_data = self.data.shape[0]
        self.name = "circle2d"
    
    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data
