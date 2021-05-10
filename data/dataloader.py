import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import *
from torch.utils.data import DataLoader, Dataset
from utils.transcript_utils import *


class RnnDataset(Dataset):
    def __init__(self, x_path, y_path, letter2index, is_train=False):
        self.is_train = is_train
        self.X = np.load(x_path, allow_pickle=True)

        Y = np.load(y_path, allow_pickle=True, encoding='bytes')
        self.Y = transform_letter_to_index(Y, letter2index)
        self.transforms = nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index].astype(np.float32)
        x = torch.tensor(x)

        if self.is_train:
            x = self.transforms(x)

        y = torch.LongTensor(self.Y[index])

        return x, y

def collate_train_val(batch):
    """
    Return:
        pad_x: the padded x (training/validation speech data)
        pad_y: the padded y (text labels - transcripts)
        x_len: the length of x
        y_len: the length of y
    """
    data = [(item[0]) for item in batch]
    data_lengths = torch.LongTensor([len(seq) for seq in data])
    data = pad_sequence(data, padding_value=0)

    max_seqlength = data.shape[0]
    input_len_ratio = torch.FloatTensor([item[0].shape[0]/float(max_seqlength) for item in batch])

    target = [(item[1]) for item in batch]
    target_lengths = torch.LongTensor([len(seq) for seq in target])
    target = pad_sequence(target, batch_first=True, padding_value=33)

    return data, target, data_lengths, target_lengths, input_len_ratio

def collate_test(batch):
    """
    Return:
        pad_x: the padded x (testing speech data)
        x_len: the length of x
    """
    data = [(item[0]) for item in batch]
    data_lengths = torch.LongTensor([len(seq) for seq in data])
    data = pad_sequence(data)

    return data, data_lengths


def get_dataloaders(train_filename, train_transcripts_filename, dev_filename, dev_transcripts_filename, letter2index, hparams):

    # # Training dataloader
    train_data = RnnDataset(train_filename, train_transcripts_filename, letter2index, is_train=True)
    train_args = dict(shuffle=True, batch_size=hparams["batch_size"], num_workers=hparams["num_workers"], drop_last=True,
                      collate_fn=collate_train_val)
    train_loader = DataLoader(train_data, **train_args)

    # # Validation dataloader
    val_data = RnnDataset(dev_filename, dev_transcripts_filename, letter2index, is_train=False)
    val_args = dict(shuffle=True, batch_size=hparams["batch_size"], num_workers=hparams["num_workers"], drop_last=True,
                    collate_fn=collate_train_val)
    val_loader = DataLoader(val_data, **val_args)

    return train_loader, val_loader