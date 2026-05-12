import torch
import os
from torch.utils import data
import numpy as np


class TrainDataset(data.Dataset):
    def __init__(self, positive, dataType, device, task, _type, algorithm_num, scene_num):
        self.device = device
        self.algorithm_num = algorithm_num
        self.scene_num = scene_num
        if positive:
            midfix = "positive_data"
        else:
            midfix = "data"
        dataset_dir = f'/data/{midfix}/{task}/train_algnum_{algorithm_num}_scenum_{scene_num}_{dataType}_{_type}.txt'
        if os.path.exists(dataset_dir):
            dataset = np.loadtxt(dataset_dir)
            self.Data = torch.from_numpy(dataset[:,:2]).int()
            self.Label = torch.from_numpy(dataset[:,-1]).float().reshape(-1,1)

    def __getitem__(self, index):
        pair=self.Data[index].to(device=self.device)
        label=self.Label[index].to(device=self.device)
        return pair, label

    def __len__(self):
        return len(self.Data)

class FinetuningDataset(data.Dataset):
    def __init__(self, positive, dataType, device, task, _type, algorithm_ind, scene_num):
        self.device = device
        self.algorithm_ind = algorithm_ind
        self.scene_num = scene_num
        if positive:
            midfix = "positive_data"
        else:
            midfix = "data"
        dataset_dir = f'/data/{midfix}/{task}/fine_algind_{algorithm_ind}_scenum_{scene_num}_{dataType}_{_type}.txt'
        if os.path.exists(dataset_dir):
            dataset = np.loadtxt(dataset_dir)
            self.Data = torch.from_numpy(dataset[:,:2]).int()
            self.Label = torch.from_numpy(dataset[:,-1]).float().reshape(-1,1)

    def __getitem__(self, index):
        pair = self.Data[index].to(device=self.device)
        label = self.Label[index].to(device=self.device)
        return pair, label

    def __len__(self):
        return len(self.Data)

class TestDataset(data.Dataset):
    def __init__(self, positive, dataType, device, _type, task, algorithm_ind, algorithm_num, scene_num):
        self.device = device
        self.algorithm_num = algorithm_num
        self.algorithm_ind = algorithm_ind
        self.scene_num = scene_num
        if positive:
            midfix = "positive_data"
        else:
            midfix = "data"
        dataset_dir = f'/data/{midfix}/{task}/test_algind_{algorithm_ind}_algnum_{algorithm_num}_scenum_{scene_num}_{dataType}_{_type}.txt'
        if os.path.exists(dataset_dir):
            dataset = np.loadtxt(dataset_dir)
            self.Data = torch.from_numpy(dataset[:,:2]).int()
            self.Label = torch.from_numpy(dataset[:,-1]).float().reshape(-1,1)

    def __getitem__(self, index):
        pair = self.Data[index].to(device=self.device)
        label = self.Label[index].to(device=self.device)
        return pair, label

    def __len__(self):
        return len(self.Data)
