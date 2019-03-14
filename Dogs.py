import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Dogs(Dataset):
    """
    A customized data loader for Dogs.
    """

    def __init__(self,
                 labelFile,
                 transform=None,
                 preload=False):

        self.images = None
        self.labels = []
        self.filenames = []
        self.labelFile = labelFile
        self.transform = transform

        with open(self.labelFile, newline='') as csvfile:
            labels = csv.reader(csvfile)
            iterLabel = iter(labels)
            next(iterLabel)  # do not include the header
            for row in iterLabel:
                self.filenames.append(row[0])
                self.labels.append(row[1])

        # if preload dataset into memory
        if preload:
            self._preload()

        self.len = len(self.filenames)

    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:
            # load images
            image = Image.open(image_fn)
            # avoid too many opened files bug
            self.images.append(image.copy())
            image.close()
            self.labels.append(label)

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)

        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return image and label
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
