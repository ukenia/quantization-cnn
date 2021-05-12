import os
import argparse
from model import ConvBlock, Wav2Letter
import torch
import time

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import *
from torch.utils.data import DataLoader, Dataset
from utils.transcript_utils import *

import os
import sys
import numpy as np
import pandas as pd
import torch
import timeit
import torch.nn as nn
import onnxruntime as rt
from torch.autograd import Variable
from ctcdecode import CTCBeamDecoder

from data.dataloader import *
from hyperparameters import get_hyperparameters
from utils.transcript_utils import *
from utils.train_utils import *
from utils.metric_utils import *
from model import Wav2Letter


parser = argparse.ArgumentParser(description='Script to evaluate a pytorch model on HW4 data')
parser.add_argument('--model', type=str, help='path to the model file')
parser.add_argument('--beamwidth', type=int, default=20, help='beamwidth for the CTCBeamDecoder')
parser.add_argument('--outvocab', type=int, default=35, help='Out vocab for the model')
parser.add_argument('--batchsize', type=int, default=1, help='Batch size to evaluate the model')
parser.add_argument('--datadir', type=str, default='data',
                    help='data directory where train.npy, dev.npy, train_transcripts.npy, and dev_transcripts.npy are present')
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda or cpu are valid options')
parser.add_argument('--quantized', action='store_true', default=False)



# Evaluate the model
@torch.no_grad()
def evaluate_model(val_loader, model, criterion, batchsize=32, beamwidth=20, device=None, quantized=False):
    predictions = []
    actuals = []
    val_loss = 0

    # distances = []
    input_name = ""
    # Set model in validation mode
    if not quantized:
        model.eval()
    else:
        input_name = model.get_inputs()[0].name

    stime = time.time()

    for i, (inputs, targets, input_lengths, target_lengths, input_len_ratio) in enumerate(val_loader):
        inputs = inputs.transpose(0, 1).transpose(1, 2).reshape((batchsize, 40, -1)).to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        # evaluate the model on the validation set
        if not quantized:
            out = model(inputs.float())
        else:
            out = torch.tensor(model.run(None, {input_name: inputs.numpy().astype(np.float32)})[0])
           
        out = out.permute(2, 1, 0)

        seq_length = out.size(0)
        out_lengths = Variable(input_len_ratio.mul_(int(seq_length)).int(), requires_grad=False)

        # Calculate validation loss
        loss = criterion(out, targets, out_lengths, target_lengths)

        actual = targets.cpu().numpy()
        actual_lengths = target_lengths.cpu().numpy()

        decoder = CTCBeamDecoder(LETTER_LIST + ["&"], beam_width=beamwidth, log_probs_input=True, blank_id=len(LETTER_LIST))
        out, _, _, out_lengths = decoder.decode(out.transpose(0, 1), out_lengths)

        # reshape for stacking
        actual = unpad_beam1(actual, actual_lengths)
        out = unpad(torch.squeeze(out, 1).cpu().numpy(), torch.squeeze(out_lengths, 1).cpu().numpy())

        # store
        predictions.append(out)
        actuals.append(actual)

        if i%50==0:
            print(f"\tBatch: {i}/{len(val_loader)}\t Loss: {loss.item():3.3f}\ttime-elapsed: {time.time()-stime:3.3f} Secs")

        del out
        del out_lengths
        del targets
        del target_lengths
        del inputs
        del input_lengths
        del input_len_ratio

        torch.cuda.empty_cache()
        val_loss += loss.item()

    p = generate_all_outputs(predictions)
    a = generate_all_outputs(actuals)

    # Calculate Validation Levenshtein distance
    if len(p) != len(a):
        print("ERROR")

    dist = calc_levenshtein_dist(a, p)
    val_loss /= len(val_loader)
    print(f"Evaluation done in {time.time()-stime: 3.3f} Secs")
    print(f"\rAverage Loss: {val_loss:3.3f} and Average Edit Distance: {dist:3.3f}")
    return dist, val_loss

if __name__=="__main__":
    LETTER_LIST = ['<', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ', '>']

    letter2index, index2letter = create_dictionaries(LETTER_LIST)
    num_workers = get_num_workers()
    hparams = get_hyperparameters()
    hparams["num_workers"] = num_workers

    

    args = parser.parse_args()
    print(args)
    print(f"Loading model: {args.model}")

    DATA_PATH = args.datadir
    DEVICE = torch.device(args.device)

    model = Wav2Letter(args.outvocab)
    
    quantized = args.quantized

    if not quantized: 
        mdl = torch.load(args.model, map_location=DEVICE)
        model.load_state_dict(mdl.state_dict())
        model.to(DEVICE)
    else:
        model = rt.InferenceSession(args.model)
    


    # Filepaths
    dev_filename = os.path.join(DATA_PATH, "dev.npy")
    dev_transcripts_filename = os.path.join(DATA_PATH, "dev_transcripts.npy")

    # criterion
    criterion = nn.CTCLoss().to(DEVICE)

    # # Validation dataloader
    val_data = RnnDataset(dev_filename, dev_transcripts_filename, letter2index, is_train=False)
    val_args = dict(shuffle=False, batch_size=args.batchsize, num_workers=hparams["num_workers"], drop_last=True,
                    collate_fn=collate_train_val)
    val_loader = DataLoader(val_data, **val_args)
    evaluate_model(val_loader, model, criterion, args.batchsize, args.beamwidth, DEVICE, quantized)

