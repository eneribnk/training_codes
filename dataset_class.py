import torch
#import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
print("successfully imported packages")


class MergedDataset(Dataset):
    def __init__(self, power, pressure):
        # data loading
        rowdata = np.loadtxt('merged_1-2116.csv', delimiter=";", dtype=np.float64, skiprows=1)
        self.power = torch.from_numpy(rowdata[:, [0]])  # if wrote like this [:,1:] all data except second column
        self.pressure = torch.from_numpy(rowdata[:, [1]])
        self.n_samples = rowdata.shape

    def __getitem__(self, index):
        # this will allow for indexing later
        return self.power[index], self.pressure[index]  # these will give tuples

    def __len__(self):
        # this allows to call len(dataset) for lenght
        return self.n_samples


dataset = MergedDataset()
#first_data = dataset_[1]
#features, labels = first_data
#print(features, labels)

# now we can see how a dataloader is used
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil((total_samples)/4)
print(total_samples, n_iterations)


