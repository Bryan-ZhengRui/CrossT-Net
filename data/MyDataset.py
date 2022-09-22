from torchvision import transforms as T
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import glob
import torch


class MyDataset(Dataset):
    def __init__(self, depth_all1, intensity_all1, normal_all1, depth_all2, intensity_all2, normal_all2, overlaps_all):
        super(MyDataset, self).__init__()
        """
        :param depth_all1: 路径列表
        :param intensity_all1: 路径列表
        :param normal_all1: 路径列表
        :param depth_all2: 路径列表
        :param intensity_all2: 路径列表
        :param normal_all2: 路径列表
        :param overlaps_all: overlap value列表
        """
        self.depth_all1 = depth_all1
        self.depth_all2 = depth_all2
        self.intensity_all1 = intensity_all1
        self.intensity_all2 = intensity_all2
        self.normal_all1 = normal_all1
        self.normal_all2 = normal_all2
        self.overlaps_all = overlaps_all

    def __getitem__(self, index):
        overlap = self.overlaps_all[index]
        depth1_path = self.depth_all1[index]
        depth2_path = self.depth_all2[index]
        intensity1_path = self.intensity_all1[index]
        intensity2_path = self.intensity_all2[index]
        normal1_path = self.normal_all1[index]
        normal2_path = self.normal_all2[index]

        depth1 = np.load(depth1_path)
        depth2 = np.load(depth2_path)
        intensity1 = np.load(intensity1_path)
        intensity2 = np.load(intensity2_path)
        normal1 = np.load(normal1_path)
        normal2 = np.load(normal2_path)

        maps1 = np.concatenate((depth1, intensity1, normal1), axis=0)
        maps2 = np.concatenate((depth2, intensity2, normal2), axis=0)

        maps1 = torch.tensor(maps1)
        maps2 = torch.tensor(maps2)
        overlap = torch.tensor(overlap)
        overlap.unsqueeze_(-1)


        return maps1,maps2, overlap

    def __len__(self):

        return len(self.overlaps_all)


class MyDataset2(Dataset):
    def __init__(self, depth_all1, intensity_all1, depth_all2, intensity_all2, overlaps_all):
        super(MyDataset2, self).__init__()
        """
        :param depth_all1: 路径列表
        :param intensity_all1: 路径列表
        :param normal_all1: 路径列表
        :param depth_all2: 路径列表
        :param intensity_all2: 路径列表
        :param normal_all2: 路径列表
        :param overlaps_all: overlap value列表
        """
        self.depth_all1 = depth_all1
        self.depth_all2 = depth_all2
        self.intensity_all1 = intensity_all1
        self.intensity_all2 = intensity_all2
        self.overlaps_all = overlaps_all

    def __getitem__(self, index):
        overlap = self.overlaps_all[index]
        depth1_path = self.depth_all1[index]
        depth2_path = self.depth_all2[index]
        intensity1_path = self.intensity_all1[index]
        intensity2_path = self.intensity_all2[index]

        depth1 = np.load(depth1_path)
        depth2 = np.load(depth2_path)
        intensity1 = np.load(intensity1_path)
        intensity2 = np.load(intensity2_path)

        maps1 = np.concatenate((depth1, intensity1), axis=0)
        maps2 = np.concatenate((depth2, intensity2), axis=0)

        maps1 = torch.tensor(maps1)
        maps2 = torch.tensor(maps2)
        overlap = torch.tensor(overlap)
        overlap.unsqueeze_(-1)


        return maps1,maps2, overlap

    def __len__(self):

        return len(self.overlaps_all)

class MyDataset3(Dataset):
    def __init__(self, depth_all1,  depth_all2, overlaps_all):
        super(MyDataset3, self).__init__()
        """
        :param depth_all1: 路径列表
        :param intensity_all1: 路径列表
        :param normal_all1: 路径列表
        :param depth_all2: 路径列表
        :param intensity_all2: 路径列表
        :param normal_all2: 路径列表
        :param overlaps_all: overlap value列表
        """
        self.depth_all1 = depth_all1
        self.depth_all2 = depth_all2
        self.overlaps_all = overlaps_all

    def __getitem__(self, index):
        overlap = self.overlaps_all[index]
        depth1_path = self.depth_all1[index]
        depth2_path = self.depth_all2[index]

        depth1 = np.load(depth1_path)
        depth2 = np.load(depth2_path)

        maps1 = torch.tensor(depth1)
        maps2 = torch.tensor(depth2)
        overlap = torch.tensor(overlap)
        overlap.unsqueeze_(-1)

        return maps1, maps2, overlap

    def __len__(self):
        return len(self.overlaps_all)