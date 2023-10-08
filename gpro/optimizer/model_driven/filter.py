import os, sys
import math
import torch
import random
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

from gpro.utils.base import open_fa, write_profile, write_fa
from gpro.generator.wgan.common import onehot2seq

class Filter():
    def __init__(self,
                 generator,
                 predictor,
                 generator_modelpath,
                 predictor_modelpath,
                 natural_datapath,
                 
                 sample_number=1000,
                 savepath=None,
                 ):
        
        self.Seqs = open_fa(natural_datapath)
        
        # define the Generator and Predictor
        self.generator = generator
        self.generator_modelpath = generator_modelpath
        
        
        self.predictor = predictor
        self.predictor_modelpath = predictor_modelpath
        
        self.natural_datapath = natural_datapath
        self.sample_number = sample_number
        self.savepath = savepath

    def seq2oh(self, seq):
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
        data = np.zeros((len(seq), self.seq_len, 4))
        data = np.float32(data)
        i = 0
        while i < len(seq):
            j = 0
            while j < len(seq[0]):
                data[i,j,:] = promoter_onehot[i][j]
                j = j + 1
            i = i + 1
        return data

    def forward(self, seed):
        seqs = self.generator.generate(self.generator_modelpath, self.sample_number, seed=seed)
        pred = self.predictor.predict_input(self.predictor_modelpath, seqs, mode="data")
        return seqs, pred
    
    def evaluate(self, seqs):
        pred = self.predictor.predict_input(self.predictor_modelpath, seqs, mode="data")
        return pred
    
    def run(self,
            MaxEpoch = 10,
            MaxPoolsize=2000):
        
        self.outdir = self.savepath
        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)
        
        seqs_list = []
        pred_list = []
        
        for epoch in range(MaxEpoch):
            seqs, pred = self.forward(epoch)
            seqs_list += seqs
            pred_list += pred

        idx = sorted(range(len(pred_list)), key=lambda k: pred_list[k], reverse=True)
        seqs_res = np.array(seqs_list)[idx]
        pred_res = np.array(pred_list)[idx]
        
        seqs_res = seqs_res[0:MaxPoolsize]
        pred_res = pred_res[0:MaxPoolsize]
        
        write_fa(self.outdir + "/ExpIter" + ".txt", seqs_res)
        write_profile(self.outdir + "/ExpIter" + ".csv", seqs_res, pred_res)
        
        pdf = PdfPages(self.outdir + '/compared_with_natural.pdf')
        plt.figure()
        
        if len(self.Seqs) > MaxPoolsize:
            self.Seqs = random.sample(self.Seqs, MaxPoolsize)
        nat_score = self.evaluate(self.Seqs)
        plt.boxplot([nat_score, pred_res])
        plt.ylabel('Score')
        plt.xticks([1,2],['Natural','Optimized'])
        pdf.savefig()
        pdf.close()
        return




