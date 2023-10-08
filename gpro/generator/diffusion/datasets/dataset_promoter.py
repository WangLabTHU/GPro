import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset
from .vocab import Vocab


class PromoterDataset(Dataset):

    # Data: sequence_data.txt
    def __init__(self, file, seq_len=50, split='train'):
        assert split in {'train', 'valid', 'test'}
        
        file_list = file.split("/")
        file_list.pop()
        self.root = '/'.join(file_list)
        path = file
        
        self.seq_len = seq_len
        self.split = split

        if not os.path.exists(self.root):
            raise RuntimeError('Dataset not found. You can use download=True to download it.')

        lines = []
        with open(path, 'r') as f:
            for line in f:
                line = line[:-1]
                lines.append(line)
        self.rawdata = lines

        # Get vocabulary
        self.vocab = Vocab()
        vocab_file = os.path.join(self.root, 'vocab.json')
        if os.path.exists(vocab_file):
            self.vocab.load_json(self.root)
        else:
            stoi = self._create_stoi()
            self.vocab.fill(stoi)
            self.vocab.save_json(self.root)

        # Preprocess data
        if not os.path.exists(self.processed_file(split)):
            self._preprocess_data(split)

        # Load data
        self.data = torch.load(self.processed_file(split))

    def __getitem__(self, index):
        return self.data[index], self.seq_len

    def __len__(self):
        return len(self.data)

    def _create_stoi(self):
        tmp = ''.join(self.rawdata)
        s = sorted(list(set(tmp)))
        stoi = {s[i]: i for i in range(len(s))}
        return stoi

    def _preprocess_data(self, split):
        
        bound1 = int(len(self.rawdata) * 0.9)
        bound2 = int(len(self.rawdata) * 0.95)
        if split == 'train':
            rawdata = self.rawdata[:bound1]
        elif split == 'valid':
            rawdata = self.rawdata[bound1:bound2]
        elif split == 'test':
            rawdata = self.rawdata[bound2:]

        # Encode characters
        tmp = ''.join(rawdata)
        data = torch.tensor([self.vocab.stoi[s] for s in tmp])
        data = data.reshape(-1, self.seq_len)

        # Save processed data
        torch.save(data, self.processed_file(split))

    # @property
    def processed_file(self, split):
        return os.path.join(self.root, 'processed_{}.pt'.format(split))
