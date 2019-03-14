#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:52:05 2019

@author: dore
"""

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

class MNIST(Dataset):
    """
    A customized data loader for MNIST.
    """
    def __init__(self,
                 root,
                 transform=None,
                 preload=False):
        """ Intialize the MNIST dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        Vars:
            - images: an array of Images
            - labels: an array of the labels (1-10) of the images
            - len: the length of all the filenames
        """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(10):
            filenames = glob.glob(osp.join(root, str(i), '*.png'))
            # osp - "root/i", glob - all of the pathnames that match "*.png"
            for fn in filenames:
                self.filenames.append((fn, i))  # (filename, label) pair
                # creates an array of tuples - filename and their label
                
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


# Create the MNIST dataset. 
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
trainset = MNIST(
    root='mnist_png/training',
    preload=True, transform=transforms.ToTensor(),
)
# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)

# load the testset
testset = MNIST(
    root='mnist_png/testing',
    preload=True, transform=transforms.ToTensor(),
)
# Use the torch dataloader to iterate through the dataset
testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

# DataLoader - iter(trainset_loader) iterate over batches (of size batch_size)
#              that are pairs of image and label

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device(cuda if use_cuda else "cpu")


# class Net is a neural net module. To use it we need to initialize the different layers (with or without paramaters)
# and then create the forward function (no need for backwards, autograd does that).
class Net(nn.Module):
    """

    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        Applies a 2D convolution over an input signal composed of several input planes.
        In the simplest case, the output value of the layer with input size (N, C_in, H_in, W_in) and output
        (N, C_out, H_out, W_out) can be precisely described as:
        out(N_i, C_out_j) = bias(C_out_j) + sum(weight(C_out_j, k) * input(N_i,k)) for k = 0:C_in-1. * here is conv
    Parameters:
        - in_channels: Number of channels in the input image.
        - out_channels: Number of channels produced by the convolution.
        - kernel_size: Size of the convolving kernel.
    More Info @ https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d

    torch.nn.Dropout2d(p=0.5, inplace=False)
        Randomly zero out entire channels (a channel is a 2D feature map, e.g., the j-th channel of the i-th sample
        in the batched input is a 2D tensor input[i,j] of the input tensor).
    Parameters:
        - p:  probability of an element to be zero-ed.
    More Info @ https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout2d

    torch.nn.Linear(in_features, out_features, bias=True)
        Applies a linear transformation to the incoming data: y = A'x + b
    Parameters:
        - in_features: size of each input sample
        - out_features: size of each output sample
    More Info @ https://pytorch.org/docs/stable/nn.html#torch.nn.Linear

    """
    def __init__(self):
        super(Net, self).__init__()  # to init the nn.Module super class.
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(epoch, print_interval=100):
    model.train()  # set training mode
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if iteration % print_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        test()


def test():
    model.eval()  # set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))


train(5)
