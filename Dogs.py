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
                 imageLoc,

                 size,
                 transform=None,
                 preload=False):

        self.images = None
        self.labels = []
        self.filenames = []
        self.imageLoc = imageLoc
        self.labelFile = labelFile
        self.size = size
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
        setLabel = set(self.labels)
        self.labelDic = {key: value for (key, value) in zip(list(setLabel), range(len(setLabel)))}

    def _preload(self):
        """
        Preload dataset to memory
        """
        self.images = []
        for image_fn in self.filenames:
            # load images
            image = Image.open(self.imageLoc + image_fn + '.jpg')
            # avoid too many opened files bug
            self.images.append(image.copy())
            image.close()

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
        else:
            # If on-demand data loading
            image_fn = self.filenames[index]
            image = Image.open(self.imageLoc + image_fn + '.jpg')

        label = self.labelDic[self.labels[index]]
        # resize the image
        image = transforms.functional.resize(image, (self.size, self.size))
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


trainset = Dogs(
    '/home/dore/Downloads/dog-breed-identification/labels.csv', '/home/dore/Downloads/dog-breed-identification/train/',
    28, preload=False, transform=transforms.ToTensor(),
)
# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)

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
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 200)
        self.fc2 = nn.Linear(200, 120)

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


train(2)