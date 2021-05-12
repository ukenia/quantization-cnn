import os
import sys
import numpy as np
import pandas as pd
import torch
import timeit
import torch.nn as nn
from torch.autograd import Variable
from ctcdecode import CTCBeamDecoder

from data.dataloader import get_dataloaders
from hyperparameters import get_hyperparameters
from utils.transcript_utils import *
from utils.train_utils import *
from utils.metric_utils import *
from model import Wav2Letter


# Train the model

def train_model(train_loader, model, hparams, criterion, optimizer):
    training_loss = 0

    # Set model in 'Training mode'
    model.train()

    # enumerate mini batches
    for i, (inputs, targets, input_lengths, target_lengths, input_len_ratio) in enumerate(train_loader):
        inputs = inputs.transpose(0, 1).transpose(1, 2).reshape((hparams["batch_size"], 40, -1)).to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        # clear the gradients
        optimizer.zero_grad()

        # compute the model output
        out = model(inputs.float())
        out = out.permute(2, 1, 0)

        seq_length = out.size(0)
        out_lengths = Variable(input_len_ratio.mul_(int(seq_length)).int(), requires_grad=False)

        # calculate loss
        loss = criterion(out, targets, out_lengths, target_lengths)

        del out
        del out_lengths
        del targets
        del target_lengths
        del inputs
        del input_lengths
        del input_len_ratio

        torch.cuda.empty_cache()

        # Backward pass
        loss.backward()

        # Update model weights
        optimizer.step()

        training_loss += loss.item()
    training_loss /= len(train_loader)
    return training_loss


# Evaluate the model
@torch.no_grad()
def evaluate_model(val_loader, model, hparams, criterion):
    predictions = []
    actuals = []
    val_loss = 0

    # distances = []

    # Set model in validation mode
    model.eval()

    for i, (inputs, targets, input_lengths, target_lengths, input_len_ratio) in enumerate(val_loader):
        inputs = inputs.transpose(0, 1).transpose(1, 2).reshape((hparams["batch_size"], 40, -1)).to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        # evaluate the model on the validation set
        out = model(inputs.float())
        out = out.permute(2, 1, 0)

        seq_length = out.size(0)
        out_lengths = Variable(input_len_ratio.mul_(int(seq_length)).int(), requires_grad=False)

        # Calculate validation loss
        loss = criterion(out, targets, out_lengths, target_lengths)

        actual = targets.cpu().numpy()
        actual_lengths = target_lengths.cpu().numpy()

        decoder = CTCBeamDecoder(LETTER_LIST + ["&"], beam_width=4, log_probs_input=True, blank_id=len(LETTER_LIST))
        out, _, _, out_lengths = decoder.decode(out.transpose(0, 1), out_lengths)

        # reshape for stacking
        actual = unpad_beam1(actual, actual_lengths)
        out = unpad(torch.squeeze(out, 1).cpu().numpy(), torch.squeeze(out_lengths, 1).cpu().numpy())

        # store
        predictions.append(out)
        actuals.append(actual)

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
    return dist, val_loss


def run_model(train_loader, val_loader, hparams, criterion, optimizer, scheduler):
    best_val_loss = 1
    for epoch in range(hparams["epochs"]):
        print("Epoch: ", epoch)

        # Train
        starttime = timeit.default_timer()
        training_loss = train_model(train_loader, model, hparams, criterion, optimizer)
        endtime = timeit.default_timer()
        print("Training time: ", (endtime - starttime) / 60)

        # Validation
        starttime = timeit.default_timer()
        val_dist, val_loss = evaluate_model(val_loader, model, hparams, criterion)
        endtime = timeit.default_timer()
        print("Validation time: ", (endtime - starttime) / 60)

        scheduler.step(val_loss)

        # Print log of accuracy and loss
        print(
            "Epoch: " + str(epoch) + ", Training loss: " + str(training_loss) + ", Validation loss: " + str(val_loss) +
            ", Validation distance: " + str(val_dist) + "\n")

        if val_loss < best_val_loss:
            torch.save(model, MODEL_PATH + "wav2letter_base_re031.pth")
            best_val_loss = val_loss


if __name__ == '__main__':
    LETTER_LIST = ['<', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ', '>']

    letter2index, index2letter = create_dictionaries(LETTER_LIST)

    BASE_PATH = ""
    DATA_PATH = os.path.join(BASE_PATH, "data-simple/")
    MODEL_PATH = os.path.join(BASE_PATH, "models/")
    PREDICTION_PATH = os.path.join(BASE_PATH, "predictions/")

    # Filepaths
    train_filename = os.path.join(DATA_PATH, "train.npy")
    train_transcripts_filename = os.path.join(DATA_PATH, "train_transcripts.npy")

    dev_filename = os.path.join(DATA_PATH, "dev.npy")
    dev_transcripts_filename = os.path.join(DATA_PATH, "dev_transcripts.npy")

    # test_filename = os.path.join(DATA_PATH, "test.npy")

    device = get_device()
    num_workers = get_num_workers()
    hparams = get_hyperparameters()
    hparams["num_workers"] = num_workers

    model = Wav2Letter(hparams["out_vocab"])
    model.to(device)

    torch.manual_seed(0)
    criterion = nn.CTCLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    train_loader, val_loader = get_dataloaders(train_filename, train_transcripts_filename, dev_filename,
                                               dev_transcripts_filename, letter2index, hparams)
    run_model(train_loader, val_loader, hparams, criterion, optimizer, scheduler)
