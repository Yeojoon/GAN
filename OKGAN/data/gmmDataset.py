import numpy as np
from torch.utils.data import Dataset, DataLoader
from .gmm import *
from .KLD_estimators import *

# Not sure how to make pytorch dataset object support infinitely
# sampling, so this just uses a finite subsample. If possible,
# use the original GaussianMixtureModel2D in gmm.py instead.
class gmmDataset(Dataset):
    def __init__(self, n_data, custom=False, tightness=3, n_clusters=4):
        gmm = (GMM2D.get_custom_model() if custom else GMM2D.get_nice_model(tightness=tightness, n_clusters=n_clusters))
        self.data = gmm.sample(num_pts=n_data)
        self._gmm = gmm
        self.out_dim = 2
        self.n_data = self.data.shape[0]
        self.n_clusters = n_clusters
        self.name = "gmm2d"

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data
    
    def mode_collapse_metric(self, fake_data):
        # samples (n, 2)
        # means (m, 2)
        real_data = self.data[:len(fake_data)]
        reverse_kl = naive_estimator(fake_data, real_data, k=10)

        return reverse_kl