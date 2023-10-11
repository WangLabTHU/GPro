import sys
import os
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker


def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def write_seq(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(data[i] + '\n')
        i = i + 1
    f.close()

def write_fa(file,data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write('>' + str(i) + '\n')
        f.write(data[i] + '\n')
        i = i + 1
    f.close()

def write_exp(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(str( np.round(data[i], 2)) + '\n')
        i = i + 1
    f.close()
    return


def write_profile(file, seqs, pred):
    path_csv = file
    res_dict = {"seqs": seqs, "pred":pred}
    df = pd.DataFrame(res_dict)
    df.to_csv(path_csv)


def freq_table_generation(samples):
    samples_num = len(samples)
    samples_length = len(samples[0])
    amino_dict = {"T":0, "C":1, "G":2, "A":3}

    table = np.zeros((4,samples_length))
    for i in range(samples_num):
        for j in range(samples_length):
            val = samples[i][j]
            val = int( amino_dict[val] )
            table[val][j] += 1

    res = np.zeros((4, samples_length))
    for k in range(samples_length):
        sum_k = np.sum(table[:,k])
        for t in range(4):
            res[t,k] = table[t,k]/sum_k
    return res

def plot_weblogos(file, seqs, plot_mode="saliency"):
    seq_len = len(seqs[0])
    alph = ["T", "C", "G", "A"]
    
    final_arr = freq_table_generation(seqs)
    
    cmap = sns.color_palette("PuBu", n_colors = 20)
    sns.set_style("whitegrid")
    
    nn_df = pd.DataFrame(data = final_arr).T
    nn_df.columns = alph
    nn_df.index = range(len(nn_df))
    nn_df = nn_df.fillna(0)
    
    font = {'size'   : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (np.minimum(seq_len / 4, 100), np.maximum(len(alph) / 3.5, 3)), dpi = 300)

    if 'saliency' in plot_mode:
        nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'weight')
    else:
        nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'probability')

    nn_logo = logomaker.Logo(nn_df, ax = ax, color_scheme = 'classic', stack_order = 'big_on_top', fade_below = 0.8, shade_below = 0.6)
    
    if 'saliency' in plot_mode:
        nn_logo.ax.set_ylabel('Weight', fontsize = 12)
    else:        
        nn_logo.ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
        nn_logo.ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'], fontsize = 12)
        nn_logo.ax.set_ylabel('Probability', fontsize = 12)

    nn_logo.style_spines(visible = False)
    nn_logo.style_spines(spines = ['left'], visible = True)
    nn_logo.ax.set_xticks(np.arange(0, seq_len, 10))
    nn_logo.ax.set_xticklabels([str(x) for x in np.arange(0, seq_len, 10)], fontsize = 12)
    nn_logo.ax.set_xlabel('Position', fontsize = 12)

    plt.tight_layout()
    
    plotnamefinal = file
    print(' Weblogo saved to ' + plotnamefinal)
    plt.savefig(plotnamefinal)
    

def data_check(datapath):
    nt_map = ["A", "T", "C", "G"]
    
    with open(datapath, 'r') as file:  
        last_line = file.readlines()[-1]  
    
    # Step0: Enter Check
    if (last_line.endswith('\n') == False):
        print("Please add an enter in the last line\n")
        return
    
    data = open_fa(datapath)
    
    # Step1: vocab detection
    tmp_string = ''.join(data)
    tmp_vocab = list(set(tmp_string))
    
    vocab_in_list = all(key in nt_map for key in tmp_vocab) 
    vocab_ood = [x for x in tmp_vocab if x not in nt_map]
    
    if (len(tmp_vocab) > 4 or vocab_in_list==False):
        print("Error character detection: ", vocab_ood)
        print("Please check if your input file contains only the four bases A, T, C, and G\n")
        return
    
    # Step2: alignment detection
    if len(set(len(x) for x in data)) > 1:
        indices = [i for i, x in enumerate(data) if len(x) not in {len(y) for y in data}] 
        print("Error alignment detection: ", indices)
        print("Please check if the line lengths of the input file are all equal\n")
        return
    
    print("Your dataset is not problematic and suitable for training.\n")
    return


