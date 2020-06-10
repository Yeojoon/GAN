import numpy as np
from torch.utils.data import Dataset, DataLoader

#from bourgan.datasets.gaussianGridDataset import gaussianGridDataset
from bourgan.datasets.SyntheticDataset import gaussianGridDataset, ringDataset, circleDataset




def getDataset(dataset_config):
    dataset_name = dataset_config['name']
    # return loader_map[dataset_name](dataset_config)
    if dataset_name == "gaussian_grid":
        return gaussianGridDataset(dataset_config['n'], dataset_config['n_data'], dataset_config['sig'])
    elif dataset_name == 'ring2d':
        return ringDataset(dataset_config['n'], dataset_config['n_data'], dataset_config['sig'], dataset_config['radius'])
    elif dataset_name == 'circle2d':
        return circleDataset(dataset_config['n_data'], dataset_config['sig'], dataset_config['radius'])
    else:
        raise ValueError("no such dataset called "+dataset_name)



# def loadGaussianGrid(dataset_config):
#     return gaussianGridDataset(dataset_config['n'], dataset_config['n_data'], dataset_config['sig'])


# loader_map = {
#     'gaussian_grid': loadGaussianGrid
# }