# purpose: Get PyTorch model summary
# date: 5th May, 2021
# comments: Default batch size 1 used

# import required libraries

import argparse
import PIL
import timeit
import os
import sys
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import *

from sklearn.metrics import accuracy_score
from torch.autograd import Variable

from datetime import datetime as dt
from model import Wav2Letter, ConvBlock

import torch
import copy
import torchsummary

# get the arguments from command line

parser = argparse.ArgumentParser()

parser.add_argument('--pypath', type=str, help='Path to the PyTorch model')
parser.add_argument('--classes', type=int, default=35,  help='Number of output classes for classification')
parser.add_argument('--batchsize', type=int, default=1,  help='Batch size for model summary')

arguments = parser.parse_args()
pytorchPath = arguments.pypath
num_classes = arguments.classes
batch_size = arguments.batchsize

def get_model_summary(model_path, classes, batch_size):
	# get model summary

    model = Wav2Letter(classes)
    model_best_ = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_best_.state_dict())

    torchsummary.summary(model, (40, 100), batch_size)

if __name__ == '__main__':
	get_model_summary(pytorchPath, num_classes, batch_size)