import h5py
#from torch.utils.data import dataloader
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, x , y ):
        self.input = x
        self.label = np.array([y])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        input = torch.Tensor(self.input[item,:,:,:])
        label = torch.Tensor(self.label[:,item])
        return input , label


class Data:
    def __init__(self,args):
        self.train_loader = None
        self.train_name   = None
        self.test_loader  = None
        self.test_name    = None
        ds = h5py.File('/data2/Public_dataset/Public_Lymphocyte_Assessment_Hackatho/test.h5', 'r')
        x = ds['x']

        organ = ds['organ']

        x = ds['x'][:]
        y = ds['y'][:]
        organ = ds['organ'][:]

        self.test_loader = data.DataLoader(
                Dataset(x ,y),
                batch_size = args.batch_size,
                shuffle = True,
                pin_memory=False,
                num_workers=args.n_threads,
            )

        self.test_name = organ