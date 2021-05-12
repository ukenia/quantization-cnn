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
from model import Wav2Letter

import onnx
import torch
import copy



def onnx_converter(model_path, converted_path):

    example_input =  torch.randn(64, 40, 100, requires_grad=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Wav2Letter(35)
    model_best_ = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_best_.state_dict())

    torch.onnx.export(model,example_input, converted_path,export_params=True,       # store the trained parameter weights inside the model file
                                    input_names=['main_input'],     # specify the name of input layer in onnx model
                                    output_names=['main_output'],
                                    dynamic_axes={'main_input' : {0 : 'batch_size', 2: 'input_length'},    # variable length axes
                                                'main_output' : {1 : 'batch_size', 2: 'output_length'}},opset_version=11)
                    

    onnx_model = onnx.load(converted_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))



def layer_names(onnx_path):
    onnx_model = onnx.load(converted_path)
    for i in range(24):
        print("Name of layer "+str(i)+" is: "+onnx_model.graph.node[i].name)