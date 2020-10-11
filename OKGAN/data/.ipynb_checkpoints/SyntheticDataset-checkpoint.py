import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import sklearn
import sklearn.datasets



class gaussianGridDataset(Dataset):
    def __init__(self, n, n_data, sig):
        self.grid = np.linspace(-4, 4, n)
        self.data = None
        self.n = n
        self.sig = sig
        for i in range(n):
            mean_x = self.grid[i]
            for j in range(n):
                mean_y = self.grid[j]
                if self.data is None:
                    self.data = np.random.multivariate_normal((mean_x, mean_y), cov=[[sig**2, 0.0], [0.0, sig**2]], size=n_data)
                else:
                    self.data = np.concatenate((self.data, np.random.multivariate_normal((mean_x, mean_y), cov=[[sig**2, 0.0], [0.0, sig**2]], size=n_data)), axis=0)

        self.out_dim = 2
        self.n_data = self.data.shape[0]
        self.name = "gaussian_grid"
    
    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data
    
    def mode_collapse_metric(self, samples):
        mean_arr = np.transpose([np.tile(self.grid, len(self.grid)), np.repeat(self.grid, len(self.grid))])
        dist_arr = np.zeros((len(samples), len(mean_arr)))

        for i, mean in enumerate(mean_arr):

            dist_arr[:, i] = np.linalg.norm(samples-mean, axis=1)

        min_ind = np.argmin(dist_arr, axis=1)
        min_dist_arr = np.amin(dist_arr, axis=1)

        ind = np.where(min_dist_arr <= self.sig*3)
        feasible_ind = min_ind[ind]
        mode_counter = Counter(feasible_ind)

        num_modes = len(mode_counter)
        real_mode_arr = np.ones(self.n**2)/self.n**2
        fake_mode_arr = np.zeros(self.n**2)
        for i in range(self.n**2):
            if i in mode_counter.keys():
                fake_mode_arr[i] = mode_counter[i]
             
        num_high_qual_samples = np.sum(fake_mode_arr)
        fake_mode_arr /= num_high_qual_samples
        reverse_kl = np.sum(np.where(fake_mode_arr != 0, fake_mode_arr * np.log(fake_mode_arr / real_mode_arr), 0))

        return num_modes, int(num_high_qual_samples), mode_counter, reverse_kl
    
    
    
class ringDataset(Dataset):
    # NB: sig is std dev
    def __init__(self, n, n_data, sig = 0.01, r = 1):
        self.data = None
        self.means = []
        self.n = n
        self.r = r
        self.sig = sig
        for i in range(n):
            mean_x = r * np.cos(i * np.pi * 2 / n)
            mean_y = r * np.sin(i * np.pi * 2 / n)
            self.means.append((mean_x, mean_y))
            data = np.random.multivariate_normal((mean_x, mean_y), cov=(sig**2 * np.eye(2)), size=n_data)
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

    def mode_collapse_metric(self, samples):
        # samples (n, 2)
        # means (m, 2)
        dist_arr = []
        for mean in self.means:
            dist_arr.append(np.linalg.norm(samples - mean, axis=1)[:,None])
        dist_arr = np.concatenate(dist_arr, axis=1) # n x m
        min_ind = np.argmin(dist_arr, axis=1)
        min_dist_arr = np.amin(dist_arr, axis=1)
        # sample is high quality if its nearest mode is < 3 std devs away
        ind = np.where(min_dist_arr <= self.sig * 3)
        feasible_ind = min_ind[ind]
        mode_counter = Counter(feasible_ind)
        
        num_modes = len(mode_counter)
        real_mode_arr = np.ones(self.n)/self.n
        fake_mode_arr = np.zeros(self.n)
        for i in range(self.n):
            if i in mode_counter.keys():
                fake_mode_arr[i] = mode_counter[i]
             
        num_high_qual_samples = np.sum(fake_mode_arr)
        fake_mode_arr /= num_high_qual_samples
        reverse_kl = np.sum(np.where(fake_mode_arr != 0, fake_mode_arr * np.log(fake_mode_arr / real_mode_arr), 0))

        return num_modes, int(num_high_qual_samples), mode_counter, reverse_kl

    
    
class nonuniform_ringDataset(Dataset):
    # NB: sig is std dev
    def __init__(self, n, n_data, sig = 0.01, r = 1, seed=None):
        self.data = None
        self.rng = np.random.RandomState(seed=seed)
        self.means = []
        self.r = r
        self.sig = sig
        self.n = n
        
        weights = [self.rng.uniform(1, 5) for _ in range(n)]
        total_weight = sum(weights)
        self.pdf = [weight / total_weight for weight in weights]
        
        for i in range(n):
            mean_x = r * np.cos(i * np.pi * 2 / n)
            mean_y = r * np.sin(i * np.pi * 2 / n)
            self.means.append((mean_x, mean_y))
        
        def sample():
            mean_idx = self.rng.choice(list(range(len(self.means))), p=self.pdf)
            mean = self.means[mean_idx]
            return self.rng.multivariate_normal(mean, cov=(sig**2 * np.eye(2)))
        
        pts = [sample() for _ in range(n_data)]
        self.data = np.array(pts)
        self.out_dim = 2
        self.n_data = self.data.shape[0]
        self.name = "nonuniform_ring2d"
    
    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data

    def mode_collapse_metric(self, samples):
        # samples (n, 2)
        # means (m, 2)
        dist_arr = []
        for mean in self.means:
            dist_arr.append(np.linalg.norm(samples - mean, axis=1)[:,None])
        dist_arr = np.concatenate(dist_arr, axis=1) # n x m
        min_ind = np.argmin(dist_arr, axis=1)
        min_dist_arr = np.amin(dist_arr, axis=1)
        # sample is high quality if its nearest mode is < 3 std devs away
        ind = np.where(min_dist_arr <= self.sig * 3)
        feasible_ind = min_ind[ind]
        mode_counter = Counter(feasible_ind)
        
        num_modes = len(mode_counter)
        real_mode_arr = np.array(self.pdf)
        fake_mode_arr = np.zeros(self.n)
        for i in range(self.n):
            if i in mode_counter.keys():
                fake_mode_arr[i] = mode_counter[i]
             
        num_high_qual_samples = np.sum(fake_mode_arr)
        fake_mode_arr /= num_high_qual_samples
        reverse_kl = np.sum(np.where(fake_mode_arr != 0, fake_mode_arr * np.log(fake_mode_arr / real_mode_arr), 0))

        return num_modes, int(num_high_qual_samples), mode_counter, reverse_kl
    
    
    
class circleDataset(Dataset):
    # NB: sig is std dev
    def __init__(self, n_data, sig = 0.05, r = 2):
        self.data = None
        self.r = r
        self.sig = sig
        n = 100
        # zero out center_frac of the points
        for i in range(n):
            mean_x = r * np.cos(i * np.pi * 2 / n)
            mean_y = r * np.sin(i * np.pi * 2 / n)
            data = np.random.multivariate_normal((mean_x, mean_y), cov=(sig**2 * np.eye(2)), size=n_data)
            if self.data is None:
                self.data = data
            else:
                self.data = np.concatenate((self.data, data), axis=0)
        data = np.random.multivariate_normal((0, 0), cov=(sig**2 * np.eye(2)), size=3*n_data)
        self.data = np.concatenate((self.data, data), axis=0)
        self.out_dim = 2
        self.n_data = self.data.shape[0]
        self.name = "circle2d"
    
    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data

    def mode_collapse_metric(self, samples):
        # note: the nearest point on the circle to any sample point is simply
        # the normalized sample point: thus, we can just take the norm of the
        # sample point and check if it falls within r +/- 3 std dev (or 0 + 3
        # std dev)
        sample_norms = np.linalg.norm(samples, axis=1)
        real_mode_arr = np.array([100.0, 3.0])/103
        fake_mode_arr = np.array([((sample_norms < self.r + 3 * self.sig) & (sample_norms > self.r - 3 * self.sig)).sum(), (sample_norms < 3 * self.sig).sum()])
        num_high_qual_samples = np.sum(fake_mode_arr)
        fake_mode_arr = fake_mode_arr.astype(float)/num_high_qual_samples
        reverse_kl = np.sum(np.where(fake_mode_arr != 0, fake_mode_arr * np.log(fake_mode_arr / real_mode_arr), 0))
        # only mode we're checking capture of is center mode
        center_captured = (sample_norms.min() < 3 * self.sig)
        return num_high_qual_samples, center_captured, reverse_kl
    
    
    
class swissrollDataset(Dataset):
    def __init__(self, n_data, noise):
        self.n_data = n_data
        self.noise = noise
        data = sklearn.datasets.make_swiss_roll(n_samples=n_data, noise=noise)[0]
        self.data = data.astype("float32")[:, [0, 2]]
        self.data /= 5
        self.out_dim = 2
        self.name = 'swissroll'
        
    def __getitem__(self, index):
        return self.data[index, :]
    
    def __len__(self):
        return self.n_data
    
    
    
class twospiralsDataset(Dataset):
    def __init__(self, n_data, sig):
        self.n_data = n_data
        self.sig = sig
        n = np.sqrt(np.random.rand(n_data // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_data // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(n_data // 2, 1) * 0.5
        self.data = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        self.data += np.random.randn(*self.data.shape) * sig
        self.out_dim = 2
        self.name = '2spirals'
        
    def __getitem__(self, index):
        return self.data[index, :]
    
    def __len__(self):
        return self.n_data
