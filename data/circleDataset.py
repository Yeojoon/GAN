import numpy as np
from torch.utils.data import Dataset, DataLoader

class circleDataset(Dataset):
    # NB: sig is actually variance, not std dev
    def __init__(self, n_data, sig = 0.06 ** 2, center_frac = 0.05, r = 1):
        angles = np.random.uniform(low=0, high=2*np.pi, size=n_data)
        sig *= (r ** 2) # scaling
        self.r = r
        self.sig = sig
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

    def format_metric(self, samples):
        num_high_quality, center_captured = self.quantitative_metric(samples)
        return {"num_high_quality" : num_high_quality,
                "center_captured" : center_captured}

    def quantitative_metric(self, samples):
        # note: the nearest point on the circle to any sample point is simply
        # the normalized sample point: thus, we can just take the norm of the
        # sample point and check if it falls within r +/- 3 std dev (or 0 + 3
        # std dev)
        sample_norms = np.linalg.norm(samples, axis=1)
        std_dev = self.sig ** 0.5
        num_high_quality = ((sample_norms < 3 * std_dev) |
            ((sample_norms < self.r + 3 * std_dev) & (sample_norms > self.r - 3 * std_dev))).sum()
        # only mode we're checking capture of is center mode
        center_captured = (sample_norms.min() < 3 * std_dev)
        return num_high_quality, center_captured

