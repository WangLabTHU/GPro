import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

def csv2fasta(csv_path, data_path, data_name):
    path = csv_path
    results = pd.read_csv(path)
    fakeB = list(results['fakeB'])
    realB = list(results['realB'])
    f2 = open(data_path + data_name + '_realB.fasta','w')
    j = 0
    for i in realB:
        f2.write('>sequence_generate_'+str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()
    f2 = open(data_path + data_name + '_fakeB.fasta','w')
    j = 0
    for i in fakeB:
        f2.write('>sequence_generate_'+str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class SeqDataset(Dataset):

    def __init__(self, path='../../deepinfomax/data/ecoli_expr.xlsx', isTrain=True, isGpu=True):
        self.path = path
        files = pd.read_csv(self.path)
        seqs = list(files['seq'])
        exprs = list(files['expr'])
        max_len = self.max_len(seqs)
        random.seed(0)
        index = list(np.arange(len(seqs)))
        random.shuffle(index)
        self.pSeq = []
        self.expr = []
        self.isTrain = isTrain
        self.split_r = 0.9
        self.isGpu = isGpu
        maxE = 1
        minE = 0
        if self.isTrain:
            start, end = 0, int(len(index)*self.split_r)
        else:
            start, end = int(len(index)*self.split_r), len(index)
        for i in range(start, end):
            if len(seqs[i]) < max_len:
                seqs[i] = seqs[i] + 'A' * (max_len - len(seqs[i]))
            self.pSeq.append(self.oneHot(seqs[i]))
            self.expr.append((exprs[i] - minE)/(maxE - minE))

    # Find the maximum length of promoter sequence
    def max_len(self, seq):
        maxLength = max(len(x) for x in seq)
        return maxLength

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
        oh = np.zeros([4, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh

    def __getitem__(self, item):
        X = self.pSeq[item][:, :]
        Z = self.expr[item]
        X = transforms.ToTensor()(X)
        X = torch.squeeze(X)
        X = X.float()
        Z = transforms.ToTensor()(np.asarray([[Z]]))
        Z = torch.squeeze(Z)
        Z = Z.float()
        if self.isGpu:
            X, Z = X.cuda(), Z.cuda()
        return {'x': X, 'z':Z}

    def __len__(self):
        return len(self.expr)





class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, stop_order='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_order = stop_order
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss
        if self.stop_order == 'min':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        elif self.stop_order == 'max':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Updation changed ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


#### 0905

def seq2onehot(seq,length):
        module = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        i = 0
        promoter_onehot = []
        while i < len(seq):
           tmp = []
           for item in seq[i]:
                if item == 't' or item == 'T':
                    tmp.append(module[0])
                elif item == 'c' or item == 'C':
                    tmp.append(module[1])
                elif item == 'g' or item == 'G':
                    tmp.append(module[2])
                elif item == 'a' or item == 'A':
                    tmp.append(module[3])
                else:
                    tmp.append([0,0,0,0])
           promoter_onehot.append(tmp)
           i = i + 1
        data = np.zeros((len(seq),length,4))
        data = np.float32(data)
        i = 0
        while i < len(seq):
            j = 0
            while j < len(seq[0]):
                data[i,j,:] = promoter_onehot[i][j]
                j = j + 1
            i = i + 1
        return data

def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def open_exp(file, operator = 'log2'):
    record = []
    f = open(file,'r')
    for item in f:
        record.append(float(item))
    max_num = max(record)
    min_num = min(record)
    result = []
    for item in record:
        if operator == 'log2':
            result.append(np.log2(item))
        elif operator == 'zero-one':
            result.append((item - min_num) / (max_num - min_num))
        else:
            result.append(item)
    f.close()
    return result