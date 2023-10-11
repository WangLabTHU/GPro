import sys
import os
import numpy as np
import pandas as pd
from .base import *


def read_fa(file_name):
    seq = []
    with open(file_name, 'r') as f:
        for item in f:
            if '>' not in item:
                seq.append(item.strip('\n'))
    return seq


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

def filter_for_experiment(seqpath, gc_strength=[0.2,0.8], poly_strength=5):
    
    seqs = open_fa(seqpath)
    res1, res2 = [], []
    
    # Step1: filter GC
    min_gc_ratio = gc_strength[0]
    max_gc_ratio = gc_strength[1]
    
    gc_res = []
    for string in seqs:
        g_count = string.count('G')  
        c_count = string.count('C')  
        
        total_count = len(string)
        gc_ratio = (g_count + c_count) / total_count
        gc_res.append(gc_ratio)
        
        if gc_ratio >= min_gc_ratio and gc_ratio <= max_gc_ratio:  
            res1.append(string)
    
    # Step2: filter poly A,T
    
    poly_res = []
    for string in seqs:
        count_A = string.count('A' * poly_strength)  
        count_T = string.count('T' * poly_strength)  
        
        poly_res.append(count_A + count_T)
        
        if count_A == 0 and count_T == 0:  
            res2.append(string)  

    seqpath_txt = os.path.splitext(seqpath)[0] + "_filter.txt"
    seqpath_csv = os.path.splitext(seqpath)[0] + "_filter.csv"
    
    res_dict = {"seqs": seqs, "gc content":gc_res, "poly at": poly_res}
    df = pd.DataFrame(res_dict)
    df.to_csv(seqpath_csv)

    res = list( set(res1)  & set(res2) )
    write_seq(seqpath_txt, res)
    
    print("We have conducted quality assessment for each sequence you provided and selected the sequences that meet your screening strength.")
    print("Results have been save separately saved in .csv and .txt file with _filter suffix")
    return