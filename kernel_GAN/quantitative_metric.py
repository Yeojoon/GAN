import numpy as np
from collections import Counter


def mode_collapse_metric(samples, n, sigma):
    grid = np.linspace(-4, 4, n)
    mean_arr = np.transpose([np.tile(grid, len(grid)), np.repeat(grid, len(grid))])
    dist_arr = np.zeros((len(samples), len(mean_arr)))
    
    for i, mean in enumerate(mean_arr):
        
        dist_arr[:, i] = np.linalg.norm(samples-mean, axis=1)
        
    min_ind = np.argmin(dist_arr, axis=1)
    min_dist_arr = np.amin(dist_arr, axis=1)
    
    ind = np.where(min_dist_arr <= sigma*3)
    feasible_ind = min_ind[ind]
    mode_counter = Counter(feasible_ind)
    
    num_modes = len(mode_counter)
    num_high_qual_samples = np.sum(np.array(list(mode_counter.values())))
    
    return num_modes, num_high_qual_samples, mode_counter
