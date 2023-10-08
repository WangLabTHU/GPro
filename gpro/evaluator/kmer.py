import os
import sys
import random
import numpy as np
import collections
from scipy.stats import pearsonr

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from ..utils.utils_evaluator import read_fa, seq2onehot

def convert(n, x, lens=4):
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
    for i in range(lens):
        res.append(0)
    res0 = []
    for i in list_b:
        res0.append(list_a[i])
    for i in range(len(res0)):
        res[lens - i - 1] = res0[len(res0) - i - 1]
    return res

def kmer_count(sequence, k_size):
    bg = ['A', 'T', 'C', 'G']
    kmer_stat = collections.OrderedDict()
    kmer_name = []
    for i in range(4**k_size):
        nameJ = ''
        cov = convert(i, 4,  lens = k_size)
        for j in range(k_size):
            nameJ += bg[cov[j]]
        kmer_name.append(nameJ)
        kmer_stat[nameJ] = 0

    if isinstance(sequence[0], str):
        for i in range(len(sequence)):
            size = len(sequence[i])
            for j in range(len(sequence[i]) - k_size + 1):
                kmer = sequence[i][j: j + k_size]
                try:
                    kmer_stat[kmer] += 1
                except KeyError:
                    kmer_stat[kmer] = 1
    else:
        for i in range(len(sequence)):
            for j in range(len(sequence[i])):
                size = len(sequence[i][j])
                if size > k_size:
                    for k in range(len(sequence[i][j]) - k_size + 1):
                        kmer = sequence[i][j][k : k + k_size]
                        try:
                            kmer_stat[kmer] += 1
                        except KeyError:
                            kmer_stat[kmer] = 1
    return kmer_stat

def get_kmer_stat(generative_seq, control_seq, k_size):
    kmer_stat_control, kmer_stat_gen = kmer_count(control_seq, k_size), kmer_count(generative_seq, k_size)
    # Normalize
    total_control = sum(kmer_stat_control.values())
    kmer_stat_control = {k: v / total_control for k, v in kmer_stat_control.items()}
    total_gen = sum(kmer_stat_gen.values())
    kmer_stat_gen = {k: v / total_gen for k, v in kmer_stat_gen.items()}
    # Get the sorted dict
    kmer_stat_control, kmer_stat_gen = collections.OrderedDict(sorted(kmer_stat_control.items())),  collections.OrderedDict(sorted(kmer_stat_gen.items()))
    return kmer_stat_control, kmer_stat_gen




def plot_kmer_with_model(generator, generator_modelpath, generator_training_datapath, report_path, file_tag, K=6, num_seqs_to_test=10000):
    ## get training_text
    
    print("Evaluation of Generator Begin.")
    print("loading training dataset from: ", generator_training_datapath)
    training_text = read_fa(generator_training_datapath)
    if num_seqs_to_test > len(training_text):
        num_seqs_to_test = len(training_text)
    
    # get samples_text
    sampling_text = generator.generate(sample_model_path=generator_modelpath, 
                                       sample_number=num_seqs_to_test, sample_output=False)
    
    # reduce computational costs
    random.shuffle(training_text)
    training_text = training_text[0:num_seqs_to_test]
    
    ## k-mer estimation
    kmer_stat_control, kmer_stat_model = get_kmer_stat(sampling_text, training_text, K)
    kmer_control_list = list(kmer_stat_control.items())
    kmer_model_list   = list(kmer_stat_model.items())
    control_mer, model_mer, control_val, model_val, ratio = [], [], [], [], []

    for i in range( pow(4,K) ):
        control_mer.append(kmer_control_list[i][0].upper())
        control_val.append(kmer_control_list[i][1])
        model_mer.append(kmer_model_list[i][0])
        model_val.append(kmer_model_list[i][1])
        if control_val[i] != 0:
            ratio.append(model_val[i]/control_val[i])
    pearsonr_val = pearsonr(model_val, control_val)[0]
    boundary = max(max(control_val), max(model_val))
    
    print("Model Pearson Correlation: {}, Boundary: {}".format(pearsonr_val, boundary) )
    
    tmp = ( str(boundary) ).split('.')[1]
    pos = len(tmp) - len(tmp.lstrip("0"))
    bound = (int(tmp[pos]) + 1) * pow(10,-pos-1)
    
    
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
    
    plt.xlim(0,bound)
    plt.ylim(0,bound)
    
    sns.regplot(x=control_val, y=model_val, color="green", scatter=False, truncate=False)
    plt.scatter(control_val,model_val, c="green", label=round(pearsonr_val,3), marker=".", s=30, alpha = 0.8, linewidths=0) # , marker=".", s=6, alpha = 0.8
    plt.legend(loc="upper left", markerscale = 1)
    ax.set_xlabel("Natural", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    
    x = np.linspace(0,bound,100)
    y = np.linspace(0,bound,100)
    plt.plot(x, y, '--', c='b', alpha = 0.6)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.linspace(0, bound, 10+1))
    ax.set_yticks(np.linspace(0, bound, 10+1))
    
    plotnamefinal = report_path + 'kmer_' + file_tag + ".png"
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    
    print('Kmer frequency plot saved to ' + plotnamefinal)
    return (pearsonr_val, plotnamefinal)


def plot_kmer(generator_training_datapath, generator_sampling_datapath, report_path, file_tag, K=6, num_seqs_to_test=10000):
    training_text = read_fa(generator_training_datapath)
    sampling_text = read_fa(generator_sampling_datapath)
    
    if num_seqs_to_test > min(len(training_text), len(sampling_text)):
        num_seqs_to_test = min(len(training_text), len(sampling_text))

    random.shuffle(training_text)
    random.shuffle(sampling_text)
    training_text = training_text[0:num_seqs_to_test]
    sampling_text = sampling_text[0:num_seqs_to_test]
    
    ## k-mer estimation
    kmer_stat_control, kmer_stat_model = get_kmer_stat(sampling_text, training_text, K)
    kmer_control_list = list(kmer_stat_control.items())
    kmer_model_list   = list(kmer_stat_model.items())
    control_mer, model_mer, control_val, model_val, ratio = [], [], [], [], []

    for i in range( pow(4,K) ):
        control_mer.append(kmer_control_list[i][0].upper())
        control_val.append(kmer_control_list[i][1])
        model_mer.append(kmer_model_list[i][0])
        model_val.append(kmer_model_list[i][1])
        if control_val[i] != 0:
            ratio.append(model_val[i]/control_val[i])
    pearsonr_val = pearsonr(model_val, control_val)[0]
    boundary = max(max(control_val), max(model_val))
    
    print("Model Pearson Correlation: {}, Boundary: {}".format(pearsonr_val, boundary) )
    
    tmp = ( str(boundary) ).split('.')[1]
    pos = len(tmp) - len(tmp.lstrip("0"))
    bound = (int(tmp[pos]) + 1) * pow(10,-pos-1)
    
    
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
    
    plt.xlim(0,bound)
    plt.ylim(0,bound)
    
    sns.regplot(x=control_val, y=model_val, color="green", scatter=False, truncate=False)
    plt.scatter(control_val,model_val, c="green", label=round(pearsonr_val,3), marker=".", s=30, alpha = 0.8, linewidths=0) # , marker=".", s=6, alpha = 0.8
    plt.legend(loc="upper left", markerscale = 1)
    ax.set_xlabel("Natural", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    
    x = np.linspace(0,bound,100)
    y = np.linspace(0,bound,100)
    plt.plot(x, y, '--', c='b', alpha = 0.6)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.linspace(0, bound, 10+1))
    ax.set_yticks(np.linspace(0, bound, 10+1))
    
    plotnamefinal = report_path + 'kmer_' + file_tag + ".png"
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    
    print('Kmer frequency plot saved to ' + plotnamefinal)
    return (pearsonr_val, plotnamefinal)



