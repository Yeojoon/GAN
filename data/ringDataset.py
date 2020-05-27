import numpy as np
from torch.utils.data import Dataset, DataLoader

class ringDataset(Dataset):
    # NB: sig is actually variance, not std dev
    def __init__(self, n, n_data, sig = 0.01 ** 2, r = 1):
        self.data = None
        self.means = []
        sig *= (r ** 2) # scaling
        self.sig = sig
        for i in range(n):
            mean_x = r * np.cos(i * np.pi * 2 / n)
            mean_y = r * np.sin(i * np.pi * 2 / n)
            self.means.append((mean_x, mean_y))
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

    def format_metric(self, samples):
        num_high_quality, modes_captured = self.quantitative_metric(samples)
        return {"num_high_quality" : num_high_quality,
                "modes_captured" : modes_captured}

    def quantitative_metric(self, samples):
        # samples (n, 2)
        # means (m, 2)
        mean_dists = []
        for mean in self.means:
            mean_dists.append(np.linalg.norm(samples - mean, axis=1)[:,None])
        mean_dists = np.concatenate(mean_dists, axis=1) # n x m
        min_indices = mean_dists.argmin(axis=1) # n x 1
        min_dists = mean_dists[np.arange(samples.shape[0]), min_indices]
        std_dev = (self.sig ** 0.5) # std dev is sqrt of variance
        # sample is high quality if its nearest mode is < 3 std devs away
        num_high_quality = (min_dists < 3 * std_dev).sum()
        # mode is designated as captured if its nearest sample is high quality
        # aka the smallest distance to that mean is < 3 std dev
        modes_captured = (mean_dists.min(axis=0) < 3 * std_dev).sum()
        return num_high_quality, modes_captured


