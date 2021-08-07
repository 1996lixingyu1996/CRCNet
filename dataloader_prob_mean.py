from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
torch.multiprocessing.set_sharing_strategy('file_system')

"""
This code is used to packing data for training or testing pytorch model
"""

class My_dataloader():
    def __init__(self, data_path, batch_size, train=True, test_size=0.2, random_seed=66):
        """

        :param data_path: list
        :param batch_size: int
        :param train: bool True / False
        :param test_size: train test ratio default 0.2
        """

        self.batch_size = batch_size
        self.test_size = test_size
        self.random_seed = random_seed

        if train:
            X_train, X_val = train_test_split(data_path, test_size=self.test_size, random_state=self.random_seed)

            train_dataset = My_dataset(list_path=X_train, train=True, transform=transforms.Compose([ToTensor()]))

            if self.batch_size == 1:
                train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=min(10, self.batch_size))

            val_dataset = My_dataset(list_path=X_val, train=False, transform=transforms.Compose([ToTensor()]))
            if self.batch_size == 1:
                val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
            else:
                val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=min(10, self.batch_size))

            self.dataloader = [train_dataloader, val_dataloader]
        else:
            test_dataset = My_dataset(list_path=data_path, train=False, transform=transforms.Compose([ToTensor()]))
            if self.batch_size == 1:
                test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
            else:
                test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=min(10, self.batch_size))

            self.dataloader = test_dataloader

    def get_loader(self):
        return self.dataloader

class My_dataset(Dataset):

    def __init__(self, list_path, train=False, transform=None):
        """

        :param list_path:
        :param train: bool
        :param transform: default None
        """

        self.list_path = list_path
        self.random = train
        self.transform = transform
        # self.label_path = label_path

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):

        data_path = self.list_path[idx]

        data = np.load(data_path)
        channels_dim = data["feature"].shape[1]
        feature = data["feature"][:, 0:channels_dim-2]  

        feature = np.swapaxes(feature, 1, 0)
        # feature = np.expand_dims(feature, 1)
        survival_time = data["survival_time"]
        survival_status = data["survival_status"]

        sample = {"feature": feature, "survival_time": survival_time, "survival_status": survival_status}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        # image, time, status = sample['feat'], sample['time'], sample['status']
        feature, survival_time, survival_status = sample["feature"], sample["survival_time"], sample["survival_status"]

        return {'feature': torch.from_numpy(feature), 'survival_time': torch.FloatTensor(survival_time), \
                'survival_status': torch.FloatTensor(survival_status)}


