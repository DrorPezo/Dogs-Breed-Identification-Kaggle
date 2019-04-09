"""
Finetuning Torchvision Models
=============================

**Original Author:** `Nathan Inkawhich <https://github.com/inkawhich>`
**Updated Version Author:** `Dore Kleinstern <https://github.com/dore42>`

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time

import os
import copy
import csv


class FineTuning():
    def __init__(self, root, model_name, num_classes, batch_size, train_var, feature_extract=True, use_pretrained=True):
        # Top level data directory. Here we assume the format of the directory conforms
        #   to the ImageFolder structure

        # root looks as the following:
        # /root/label_file - the csv file with the labels
        # /root/train_var_dir/train/ ...
        # /root/train_var_dir/var/ ... - a folder with 2 subfolders named train and var with both of them having
        #                                subfolders with the name of the label and picture samples
        # /root/test_dir/<any_folder_name>/ ... - a folder with another folder (an empty label) and inside are all the
        #                                         test pictures

        self.root = root#"/home/dore/Documents/dog-breed-identification/"

        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        self.model_name = model_name#"vgg"

        # Number of classes in the dataset
        self.num_classes = num_classes#120

        # Batch size for training (change depending on how much memory you have)
        self.batch_size = batch_size#64

        self.train_var = train_var#sort

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = feature_extract#True

        self.use_pretrained = use_pretrained#True

        # Initialize the model for this run
        self._initialize_model()

        # Data augmentation and normalization for training
        # Just normalization for validation and testing
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Create training and validation datasets
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.root + self.train_var, x), self.data_transforms[x]) for x in
                          ['train', 'val']}
        # Create training and validation dataloaders
        self.dataloaders_dict = {
            x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) \
                for x in ['train', 'val']}

        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        self.model = self.model.to(self.device)

        if feature_extract:
            self.params_to_update = []
            for _, param in self.model.named_parameters():
                if param.requires_grad:
                    self.params_to_update.append(param)
        else:
            self.params_to_update = self.model.parameters()

        # Default loss fxn
        self.criterion = nn.CrossEntropyLoss()

        # Default optimizer
        self.optimizer = optim.SGD(self.params_to_update, lr=0.001, momentum=0.9)


    def _initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        self.model = None
        self.input_size = 0

        if self.model_name == "resnet":
            """ Resnet18
            """
            self.model = models.resnet18(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            self.model = models.alexnet(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "vgg":
            """ VGG16_bn
            """
            self.model = models.vgg16(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            self.model = models.squeezenet1_0(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad()
            self.model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model.num_classes = self.num_classes
            self.input_size = 224

        elif self.model_name == "densenet":
            """ Densenet
            """
            self.model = models.densenet161(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "inception":
            """ Inception v3 
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model = models.inception_v3(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad()
            # Handle the auxilary net
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

    def _set_parameter_requires_grad(self):
        if self.feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False

    def train(self, num_epochs):
        since = time.time()
        is_inception = self.model_name == "inception"

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders_dict[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.best_loss = best_loss
        return val_acc_history

    def __get_params_to_update__(self):
        return self.params_to_update

    def __set_optim__(self, optim):
        self.optimizer = optim

    def __set_criterion__(self, criterion):
        self.criterion = criterion

    def kaggle_csv(self, test_folder, csv_file):
        class2idx = self.image_datasets['train'].class_to_idx
        images = datasets.ImageFolder(self.root + test_folder, self.data_transforms['test'])
        image_names = [x[0] for x in images.imgs]
        dataloader = torch.utils.data.DataLoader(images, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.model.eval()  # Set model to evaluate mode

        num_images = 0
        with open(self.root + csv_file, mode='w') as testCsv:
            csv_writer = csv.writer(testCsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            firstRow = ['id']
            for name in class2idx:
                firstRow.append(name)
            csv_writer.writerow(firstRow)  # Write the header of the first row
            # Iterate over data.
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)

                with torch.no_grad():
                    logits = self.model(inputs)
                    _, preds = torch.max(logits, 1)
                    batch_len = len(preds)
                    for i, fname in enumerate(image_names[num_images:(num_images + batch_len)]):
                        fileNameExt = os.path.basename(fname)
                        fileName, _ = os.path.splitext(fileNameExt)  # Just the file name w/o extenstions
                        softmax = np.exp(logits[i].cpu().detach().numpy()) / np.sum(
                            np.exp(logits[i].cpu().detach().numpy()),
                            axis=0)  # Softmax
                        strSoftmax = np.array([str(x) for x in softmax])  # Change it to string
                        csvRow = np.concatenate((np.array([fileName]), strSoftmax))  # Concatenate the string
                        csv_writer.writerow(csvRow)  # Write the next line
                    num_images += batch_len

    def save_model(self, model_dir):
        torch.save(self.model, model_dir + self.model_name + '_loss_' + str(self.best_loss) + '.pth')

    def load_model(self, model_file):
        self.model = torch.load(model_file)
        self.model.eval()
