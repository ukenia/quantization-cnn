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

from onnx_tf.backend import prepare
import tensorflow as tf


def onnx_to_tfgraph(onnx_path, converted_path):
    # load the model saved in onnx format
   
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    # prepare model for exporting to tensorFlow using tensorFlow backend
    tf_rep = prepare(model_onnx)

    tf_rep.run(example_input.detach())

    # # export tensorFlow backend to tensorflow tf file
    tf_rep.export_graph(converted_path)


def tfgraph_to_tflite(graph_path, tflite_path):

    
    # Load the SavedModel.
    saved_model_obj = tf.saved_model.load(export_dir=graph_path)

    # Load the specific concrete function from the SavedModel.
    concrete_func = saved_model_obj.signatures['serving_default']
    print(concrete_func.output_shapes)
    # Set the shape of the input in the concrete function.
    concrete_func.inputs[0].set_shape([128,40,None])

    print(concrete_func.outputs)

    # Convert the model to a TFLite model.
    example_input = torch.randn(128, 40, 140, requires_grad=True)

    f = concrete_func
    print(f(main_input=example_input.detach())['output_0'].shape)

    converter =  tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

