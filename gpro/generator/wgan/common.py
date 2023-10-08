import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset

class Resblock(nn.Module):

    def __init__(self, kernel_size, model_dim = 512):
        super(Resblock, self).__init__()
        self.model_dim = model_dim

        self.normal_way = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(self.model_dim, self.model_dim, kernel_size=kernel_size, padding=2, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv1d(self.model_dim, self.model_dim, kernel_size=kernel_size, padding=2, bias=True))

    def forward(self, inputs):
        return 0.3 * self.normal_way(inputs) + inputs


class Seq_loader(Dataset):

    def __init__(self, seq_file, seq_len = 50):
        super(Seq_loader, self).__init__()
        self.seq_file = seq_file
        self.seq_len = seq_len
        self.seq = self.read_fa(self.seq_file)
        self.onehot = self.seq_onehot(self.seq)

    def seq_onehot(self, seq):
        ref = {'T': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'A': [0, 0, 0, 1]} # MODIFIED
        onehot = []
        for item in seq:
            tmp = []
            for letter in item:
                tmp.append(ref[letter])
            onehot.append(tmp)
        return np.array(onehot)

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

def onehot2seq(onehot, seq_len = 50):
    onehot = onehot.view(-1, seq_len, 4).cpu().detach().numpy()
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


def sequence2fa(sequence, file_name):
    with open(file_name, 'w') as f:
        for item in sequence:
            f.write('>' + '\n')
            f.write(item + '\n')