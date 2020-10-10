import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from collections import Counter
from scipy.stats import poisson



class PoissonDigitDataset(Dataset):
    def __init__(self, digit_generator, n_data, mean=100):
        self.data = np.zeros((n_data, 28, 84))
        self.label = np.zeros(n_data, )
        self.n_data = n_data
        self.mean = mean
        
        for i in range(n_data):
            image, number = digit_generator(mean=mean)
            self.data[i] = image
            self.label[i] = number
        
        self.name = "poisson_digit"
    
    def __getitem__(self, index):
        return self.data[index, :]
    
    def __getlabel__(self, index):
        return self.label[index]

    def __len__(self):
        return self.n_data
    
    def poisson_pmf(self, x):
        return poisson.pmf(x, self.mean)
    
    def to_img(self, x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        return x
    
    def serialize_samples(self, samples):
        serialized_samples = torch.zeros(samples.size(0) * 3, 28, 28)
        for i in range(samples.size(0)):
            for j in range(3):
                serialized_samples[3*i+j] = samples[i, :, 28*j:28*(j+1)]
                
        if torch.cuda.is_available():
            serialized_samples = serialized_samples.cuda()
                
        return serialized_samples.view(serialized_samples.size(0), 1, 28, 28)
    
    def fake_label(self, samples, lenet):
        length_samples = len(samples)
        samples = self.serialize_samples(samples)
        gen_imgs = self.to_img(samples)
        gen_imgs_resize = nn.functional.interpolate(gen_imgs, size=(32, 32), mode='bilinear', align_corners=True).type(torch.cuda.FloatTensor)
        lenet_output = lenet(gen_imgs_resize)
        gen_index = np.argmax(lenet_output.cpu().detach().numpy(), axis = -1)
        gen_label = gen_index.reshape(-1, 3)
        
        fake_label = np.zeros(length_samples, )
        for i in range(length_samples):
            fake_label[i] = 100*gen_label[i, 0] + 10*gen_label[i, 1] + gen_label[i, 2]
            
        return fake_label
    
    def reverse_kl_div(self, samples, lenet):
        real_counter = Counter(self.label)
        print('real counter is ', real_counter)
        fake_counter = Counter(self.fake_label(samples, lenet))
        print('fake counter is ', fake_counter)
        
        #min_val = int(min(min(real_counter.keys()), min(fake_counter.keys())))
        #max_val = int(max(max(real_counter.keys()), max(fake_counter.keys())))
        #min_val = 70
        #max_val = 130
        min_val = int(min(real_counter.keys()))
        max_val = int(max(real_counter.keys()))
        
        real_label = np.arange(min_val, max_val+1)
        poisson_vec = np.vectorize(self.poisson_pmf)
        real_dist = poisson_vec(real_label)
        
        fake_label = np.zeros(max_val-min_val+1, )
        for i in range(min_val, max_val+1):
            if i in fake_counter.keys():
                fake_label[i-min_val] = fake_counter[i]
        
        fake_dist = fake_label / np.sum(fake_label)
        reverse_kl = np.sum(np.where(fake_dist != 0, fake_dist * np.log(fake_dist / real_dist), 0))
        
        return reverse_kl
        
        
    
    
        