# purpose: Function for converting QAT PyTorch model to ONNX
# date: 2nd May, 2021
# comments: Weird error still left to resolve
# reference: https://discuss.pytorch.org/t/onnx-export-of-quantized-model/76884/7

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
import io

# get the arguments from command line

parser = argparse.ArgumentParser()

parser.add_argument('--pypath', type=str, help='Path to the PyTorch model')
parser.add_argument('--outpath', type=str, default='', help='Path to the output ONNX model')
parser.add_argument('--classes', type=int, default=35,  help='Number of output classes for classification')

arguments = parser.parse_args()
pytorchPath = arguments.pypath
outPath = arguments.outpath
num_classes = arguments.classes

def export_qat(model_path, output_path, classes):
	# import the pytorch model -> declare the dynamic axes and classes -> export to onnx

	torch.backends.quantized.engine = "qnnpack"
	qconfig = torch.quantization.default_qconfig
	model = Wav2Letter(classes)
	model.qconfig = qconfig
	model = torch.quantization.prepare(model)
	model = torch.quantization.convert(model)
	model_best_ = torch.load(model_path, map_location=torch.device('cpu'))
	model.load_state_dict(model_best_)

	x_numpy = np.random.rand(64, 40, 100).astype(np.float32)
	x = torch.from_numpy(x_numpy).to(dtype=torch.float32)
	x = torch.quantize_per_tensor(x, 1.0, 0, torch.quint8)
	outputs = model(x)
	input_names = ["x"]
	#        outputs = model(x)

	traced = torch.jit.trace(model, x)
	buf = io.BytesIO()
	torch.jit.save(traced, buf)
	buf.seek(0)

	model = torch.jit.load(buf)
	torch.onnx.export(model, x, output_path, input_names=input_names, example_outputs=outputs, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9)
	print("QAT ONNX model exported to: "+output_path)

	onnx_model = onnx.load(output_path)

	return onnx_model

if __name__ == '__main__':
	export_qat(pytorchPath, outPath, num_classes)
