# purpose: Functions for converting PyTorch model to ONNX and quantizing it
# date: 2nd May, 2021
# comments: Quantize flag dictates if the model is to be quantized or just exported to ONNX
# reference: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# reference: https://www.onnxruntime.ai/docs/how-to/quantization.html

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

import onnx
from onnxruntime.quantization import quantize_dynamic,QuantType
import torch
import copy

# get the arguments from command line

parser = argparse.ArgumentParser()

parser.add_argument('--excludenodes', nargs='+', default = [], help='Nodes to be excluded from quantization')
parser.add_argument('--pypath', type=str, help='Path to the PyTorch model')
parser.add_argument('--outpath', type=str, default='', help='Path to the output ONNX model')
parser.add_argument('--quantpath', type=str, default='', help='Path to the quantized ONNX model')
parser.add_argument('--quantize', type=int, default=1,  help='Is the model to be quantized')
parser.add_argument('--classes', type=int, default=35,  help='Number of output classes for classification')

arguments = parser.parse_args()
excludeNodes = arguments.excludenodes
pytorchPath = arguments.pypath
outPath = arguments.outpath
quantPath = arguments.quantpath
quantizeFlag = arguments.quantize
num_classes = arguments.classes

def exportToOnnx(model_path, output_path, classes):
	# import the pytorch model -> declare the dynamic axes and classes -> export to onnx

    example_input =  torch.randn(64, 40, 100, requires_grad=True)
    
    model = Wav2Letter(classes)
    model_best_ = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_best_.state_dict())

    torch.onnx.export(model,example_input, output_path, export_params=True,       # store the trained parameter weights inside the model file
                                    input_names=['main_input'],     # specify the name of input layer in onnx model
                                    output_names=['main_output'],
                                    dynamic_axes={'main_input' : {0 : 'batch_size', 2: 'input_length'},    # variable length axes
                                                'main_output' : {1 : 'batch_size', 2: 'output_length'}}, opset_version=11)
                    
    onnx_model = onnx.load(output_path)
    for i in range(len(onnx_model.graph.node)):
        print("Name of layer "+str(i)+" is: "+onnx_model.graph.node[i].name)

    print("ONNX model was saved at: "+output_path)

def quantizeModel(model_path, nodes_to_exclude, output_path):
	# quantize dynamic with uint8 and excluding nodes
	quantized_model = quantize_dynamic(model_path, output_path, weight_type=QuantType.QUInt8,nodes_to_exclude=nodes_to_exclude)
	print("Quantized model saved at: "+output_path)


if __name__ == '__main__':
	if quantizeFlag == 0:
		exportToOnnx(pytorchPath, outPath, num_classes)
	else:
		quantizeModel(outPath, excludeNodes, quantPath)