import csv
from shutil import copy
from pathlib import Path
import os

labelFile = '/home/dore/Downloads/dog-breed-identification/labels.csv'
imageLoc = '/home/dore/Downloads/dog-breed-identification/train/'
train = '/home/dore/Downloads/dog-breed-identification/sort/train/'
val = '/home/dore/Downloads/dog-breed-identification/sort/val/'


with open(labelFile, newline='') as csvfile:
    labels = csv.reader(csvfile)
    iterLabel = iter(labels)
    next(iterLabel)  # do not include the header
    trainVal = 0.1

    interval = 0
    for row in iterLabel:
        if interval % int(1/trainVal) == 0:
            if not os.path.exists(val + row[1]):
                os.makedirs(val + row[1])
            copy(imageLoc + row[0] + '.jpg', val + row[1])
        else:
            if not os.path.exists(train + row[1]):
                os.makedirs(train + row[1])
            copy(imageLoc + row[0] + '.jpg', train + row[1])
        interval += 1
