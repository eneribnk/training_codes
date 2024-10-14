import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

'''
First of all, you need to load both X (features) and y's (predicted values).
Numpy is ok for data manipulation, but for Dataset loading I find Pandas to be more helpful. 
Nonetheless, you will need to provide tensors for the ML training.
Also, you need to assure the DataLoader takes the input the way it can parse it, which is the for loop I use 
at the end of the file.
I will write below the code I suggest you use, but it is not mandatory, it is formed the way I am used to writing it and 
(most importantly), how other people are used to seeing it!!
Also, please notice the notation and the names I use.
'''


class MergedDataset(Dataset):
    def __init__(self):
        # Constructor function
        df = pd.read_csv('merged_1-2116.csv', sep=';')

        X_columns = ['Power', 'Pressure']  # features
        y_columns = [col for col in df.columns if col not in X_columns]  # To be predicted

        self.X, self.y = df[X_columns], df[y_columns]

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]


dataset = MergedDataset()

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print("Data:", data)
    print("Labels:", labels)
