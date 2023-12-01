import os,sys
import numpy as np
from tqdm import tqdm
from time import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


def onehot2seq(onehot):
    onehot = onehot.permute(0,2,1).cpu().detach().numpy()
    ref = {0: 'T', 1: 'C', 2: 'G', 3: 'A'}
    seq_list = []
    for item in onehot:
        seq = ''
        for letter in item:
            idx = np.where(letter == np.amax(letter))[0]
            if ( len(idx) ==1 ):
                letter = int(idx)
            else:
                letter = np.random.choice(idx)
            seq = seq + ref[letter]
        if seq != '':
            seq_list.append(seq)
    return seq_list

'''
input should be [batch_size, seq_len, n_tokens]
'''

def seq2onehot(seq):
    ref = {'T': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'A': [0, 0, 0, 1], 'N':[1, 0, 0, 0]} # MODIFIED
    onehot = []
    for item in seq:
        tmp = []
        for letter in item:
            tmp.append(ref[letter])
        onehot.append(tmp)
    return np.array(onehot)

class Seq_loader(Dataset):

    def __init__(self, seq_file, seq_len = 80):
        super(Seq_loader, self).__init__()
        self.seq_file = seq_file
        self.seq_len = seq_len
        self.seq = self.read_fa(self.seq_file)
        self.onehot = seq2onehot(self.seq)


    def read_fa(self, file_name):
        seq = []
        with open(file_name, 'r') as f:
            for item in f:
                if '>' not in item:
                    seq.append(item[0:self.seq_len].strip('\n'))
        return seq

    def __len__(self, ):
        seq = self.read_fa(self.seq_file)
        return len(seq)

    def __getitem__(self, idx):
        sample = {'seq': self.seq[idx], 'onehot': self.onehot[idx]}
        return sample