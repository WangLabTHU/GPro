
import os,sys
import numpy as np
import pandas as pd

import time
import logging
import collections
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import squeeze
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def seq2onehot(seq):
    charmap = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros([len(charmap), len(seq)])

    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            encoded[charmap[seq[i]], i] = 1
    return encoded

def onehot2seq(onehot, seq_len = 165):
    # [4,165]
    onehot = onehot.view(-1, seq_len, 4).cpu().detach().numpy()
    ref = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq_list = []
    for item in onehot:
        seq = ''
        for letter in item:
            letter = int(np.where(letter == np.amax(letter))[0])
            seq = seq + ref[letter]
        if seq != '':
            seq_list.append(seq)
    return seq_list

class LoadData(Dataset):

    def __init__(self, path='data/ecoli_100_space_fix.csv', split_r=0.9, is_train=True, gpu_ids='0'):
        realB = list(pd.read_csv(path)['realB'])
        realA = list(pd.read_csv(path)['realA'])
        data_size = len(realB)
        split_idx = int(data_size * split_r)
        noise_dim = 128
        self.gpu_ids = gpu_ids
        if is_train:
            st, ed = 0, split_idx
        else:
            st, ed = split_idx, data_size
        self.storage, self.input_seq = [], []
        for i in range(st, ed, 1):
            self.storage.append(seq2onehot(realB[i].split('\n')[0].upper()))
            self.input_seq.append(seq2onehot(realA[i].split('\n')[0].upper()))

    def __getitem__(self, item):
        in_seq, label_seq = transforms.ToTensor()(self.input_seq[item]), transforms.ToTensor()(self.storage[item]) # [1,4,165]
        device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
        return {'in': in_seq[0, :].float().to(device), 'out': squeeze(label_seq).float().to(device)}

    def __len__(self):
        return len(self.storage)
    
    
def save_sequence(tensorSeq, tensorInput, tensorRealB, save_path='results/', name='', cut_r=0.1):
    i = 0
    results =collections.OrderedDict()
    results['fakeB'] = []
    results['realA'] = []
    results['realB'] = []
    for seqT in tensorSeq:
        label = 'fakeB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for seqT in tensorInput:
        label = 'realA'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for seqT in tensorRealB:
        label = 'realB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for label in ['realA', 'fakeB', 'realB']:
        results[label] = results[label][0 : int(cut_r * len(results[label]))]
    results = pd.DataFrame(results)
    save_name = save_path + name + time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime(time.time())) + 'results.csv'
    results.to_csv(save_name, index=False)
    return save_name


def reserve_percentage(tensorInput, tensorSeq):
    results =collections.OrderedDict()
    results['fakeB'] = []
    results['realA'] = []
    for seqT in tensorSeq:
        label = 'fakeB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
    for seqT in tensorInput:
        label = 'realA'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
    c, n = 0.0, 0.0
    for i in range(len(results['fakeB'])):
        seqA = results['realA'][i]
        seqB = results['fakeB'][i]
        for j in range(len(seqA)):
            if seqA[j] != 'M':
                n += 1
                if seqA[j] == seqB[j]:
                    c += 1
    return 100*c/n

def csv2fasta(csv_path, data_path, data_name):
    path = csv_path
    results = pd.read_csv(path)
    fakeB = list(results['fakeB'])
    realB = list(results['realB'])
    f2 = open(data_path + data_name + '_fakeB.fasta','w')
    j = 0
    for i in fakeB:
        f2.write('>sequence_generate_'+str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()
    f2 = open(data_path + data_name + '_realB.fasta', 'w')
    j = 0
    for i in realB:
        f2.write('>sequence_generate_' + str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()


def tensor2seq(input_sequence, label):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_sequence, np.ndarray):
        if isinstance(input_sequence, torch.Tensor):  # get the data from a variable
            sequence_tensor = input_sequence.data
        else:
            return input_sequence
        sequence_numpy = sequence_tensor.cpu().float().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        sequence_numpy = input_sequence
    return decode_oneHot(sequence_numpy, label)

def decode_oneHot(seq, label):
    keys = ['A', 'C', 'G', 'T']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        if label == 'realA':
            if np.max(seq[:, i]) != 1:
                dSeq += 'M'
            else:
                pos = np.argmax(seq[:, i])
                dSeq += keys[pos]
        else:
            pos = np.argmax(seq[:, i])
            dSeq += keys[pos]
    return dSeq

def get_logger(log_path='cache/training_log/', name='log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = log_path + name + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def convert(n, x):
    list_a = [0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
    list_b = []
    while True:
        s,y = divmod(n,x)
        list_b.append(y)
        if s == 0:
            break
        n = s
    list_b.reverse()
    res = []
    for i in range(x):
        res.append(0)
    res0 = []
    for i in list_b:
        res0.append(list_a[i])
    for i in range(len(res0)):
        res[x - i - 1] = res0[len(res0) - i - 1]
    return res

def kmer_frequency(valid_path, ref_path, k=4, save_path='cache/', save_name='99'):
    print('Start saving the frequency figure......')
    bg = ['A', 'C', 'G', 'T']
    valid_kmer, ref_kmer = collections.OrderedDict(), collections.OrderedDict()
    kmer_name = []
    for i in range(4**k):
        nameJ = ''
        cov = convert(i, 4)
        for j in range(k):
                nameJ += bg[cov[j]]
        kmer_name.append(nameJ)
        valid_kmer[nameJ], ref_kmer[nameJ] = 0, 0
    valid_df = pd.read_csv(valid_path)
    ref_df = pd.read_csv(ref_path)
    fakeB = list(valid_df['fakeB'])
    realB = list(ref_df['realB'])
    realA = list(ref_df['realA'])
    valid_num, ref_num = 0, 0
    for i in range(len(fakeB)):
        for j in range(len(fakeB[0]) - k + 1):
            k_mer = fakeB[i][j : j + k]
            mask_A = realA[i][j : j + k]
            if 'A' not in mask_A or 'T' not in mask_A or 'C' not in mask_A or 'G' not in mask_A:
                valid_kmer[k_mer] += 1
                valid_num += 1
    for i in range(len(realB)):
        for j in range(len(realB[0]) - k + 1):
            realB[i] = realB[i].strip()
            k_mer = realB[i][j : j + k].strip()
            if len(k_mer) == 2:
                print(realB[i])
            ref_num += 1
            ref_kmer[k_mer] += 1
    for i in kmer_name:
        ref_kmer[i], valid_kmer[i] = ref_kmer[i]/ref_num, valid_kmer[i]/valid_num
    plt.plot(list(ref_kmer.values()))
    plt.plot(list(valid_kmer.values()))
    plt.legend(['real distribution', 'model distribution'])
    plt.title('{}_mer frequency'.format(k))
    plt.xlabel('{}_mer index'.format(k))
    plt.ylabel('{}_mer frequency'.format(k))
    plt.savefig('{}_{}_{}_mer_frequency.png'.format(save_path, save_name, k))
    plt.close()
    print('Saving end!')
    
def polyAT_freq(valid_path, ref_path):
    A_dict_valid = {'AAAAA':0, 'AAAAAA':0, 'AAAAAAA': 0, 'AAAAAAAA': 0}
    A_dict_ref = {'AAAAA': 0, 'AAAAAA': 0, 'AAAAAAA': 0, 'AAAAAAAA': 0}
    T_dict_valid = {'TTTTT':0, 'TTTTTT':0, 'TTTTTTT': 0, 'TTTTTTTT': 0}
    T_dict_ref = {'TTTTT': 0, 'TTTTTT': 0, 'TTTTTTT': 0, 'TTTTTTTT': 0}
    valid_df = pd.read_csv(valid_path)
    ref_df = pd.read_csv(ref_path)
    fakeB = list(valid_df['fakeB'])
    realB = list(ref_df['realB'])
    for i in range(len(fakeB)):
        fakeBt = fakeB[i]
        for keys in A_dict_valid.keys():
            for j in range(0, len(fakeBt) - len(keys) + 1):
                if fakeBt[j : j + len(keys)] == keys:
                    A_dict_valid[keys] += 1
        for keys in T_dict_valid.keys():
            for j in range(0, len(fakeBt) - len(keys) + 1):
                if fakeBt[j : j + len(keys)] == keys:
                    T_dict_valid[keys] += 1

    for i in range(len(realB)):
        realBt = realB[i]
        for keys in A_dict_ref.keys():
            for j in range(0, len(realBt) - len(keys) + 1):
                if realBt[j : j + len(keys)] == keys:
                    A_dict_ref[keys] += 1
        for keys in T_dict_ref.keys():
            for j in range(0, len(realBt) - len(keys) + 1):
                if realBt[j : j + len(keys)] == keys:
                    T_dict_ref[keys] += 1

    for keys in A_dict_valid.keys():
        A_dict_valid[keys] = A_dict_valid[keys] / len(fakeB)
        A_dict_ref[keys] = A_dict_ref[keys] / len(realB)
    for keys in T_dict_valid.keys():
        T_dict_valid[keys] = T_dict_valid[keys] / len(fakeB)
        T_dict_ref[keys] = T_dict_ref[keys]/len(realB)

    return A_dict_valid, A_dict_ref, T_dict_valid, T_dict_ref


def make_dir(model_name, savepath):
        
    cache_path = os.path.join(savepath, 'cache/' + model_name)
    cache_figure_path = "{}/figure/".format(cache_path)
    cache_training_log_path = "{}/training_log/".format(cache_path)
    cache_gen_iter_path = "{}/gen_iter/".format(cache_path)
    cache_inducible_path = "{}/inducible/".format(cache_path)
    
    if not os.path.exists(cache_figure_path):
        os.makedirs(cache_figure_path)
    if not os.path.exists(cache_training_log_path):
        os.makedirs(cache_training_log_path)
    if not os.path.exists(cache_gen_iter_path):
        os.makedirs(cache_gen_iter_path)
    if not os.path.exists(cache_inducible_path):
        os.makedirs(cache_inducible_path)
    
    check_path = os.path.join(savepath, 'check/' + model_name)
    
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    
    return cache_figure_path, cache_training_log_path, cache_gen_iter_path, cache_inducible_path, check_path