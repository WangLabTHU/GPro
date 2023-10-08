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
from gpro.generator.wgan.wgan import Generator
from gpro.predictor.cnn_k15.cnn_k15 import CNN_K15

from tqdm import tqdm

matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")


# https://blog.csdn.net/youcans/article/details/116371656

class AnnealingAlgorithm():
    def __init__(self,
                 generator_modelpath,
                 predictor_modelpath,
                 natural_datapath,
                 seed = 0,
                
                 savepath = None,
                 z_dim = 128,
                 seq_len = 50
                 ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Seqs = open_fa(natural_datapath)
        random.seed(seed)
        
        self.generator = Generator(seq_len=seq_len).to(self.device)
        state_dict = torch.load(generator_modelpath)
        self.generator.load_state_dict(state_dict['model'])
        self.generator.eval()
        
        self.predictor = CNN_K15(input_length=seq_len).to(self.device)
        state_dict = torch.load(predictor_modelpath)
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        
        self.savepath = savepath
        self.z_dim = z_dim
        self.seq_len = seq_len
    
    
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
    
    def sample(self, z):
        noise = torch.tensor(z, dtype=float)
        noise = noise.to(torch.float32).to(self.device)
        outputs = self.generator(noise)
        outputs = outputs.permute(0, 2, 1)
        seqs = onehot2seq(outputs, self.seq_len)
        return np.array(seqs)
    
    def evaluate(self, seqs):
        onehot = self.seq2oh(seqs)
        onehot = torch.tensor(onehot, dtype=float)
        onehot = onehot.to(torch.float32).to(self.device).permute(0,2,1)
        outputs = self.predictor(onehot)
        pred = outputs.flatten().tolist()
        return np.array(pred)
    
    def run(self,
            MaxPoolsize=2000,
            alpha=0.98,
            tInital=1,
            tFinal=1e-2,
            meanMarkov=100,
            scale=0.5,
            MaxIter=1000):
        
        
        self.outdir = self.savepath
        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)

        self.x = np.random.normal( size=(self.z_dim) )
        self.pred = self.evaluate(self.sample(self.x))
        
        totalMar = 0
        totalImprove = 0
        kIter = 0
        xBest = self.x
        bestscore = self.pred
        tNow = tInital
        recordPredNow = []
        recordPredBest = []
        recordPBad = []
        self.scale = scale
        self.xMin = [-10] * len(self.x)
        self.xMax = [10] * len(self.x)
        
        seqs_list = []
        pred_list = []
        
        # Outer circulation
        while tNow >= tFinal:
            kBetter = 0
            kBadAccept = 0
            kBadRefuse = 0
            
            # Inner circulation
            for k in range(meanMarkov):
                totalMar += 1
                x_new = self.Step(self.x)
                pred_new = self.evaluate(self.sample(x_new))
                deltaE = pred_new - self.pred

                if deltaE>0:
                    accept = True
                    kBetter += 1
                else:
                    pAccept = math.exp(deltaE / tNow)
                    if pAccept > random.random():
                        accept = True
                        kBadAccept += 1
                    else:
                        accept = False
                        kBadRefuse += 1
            
                # storing
                if accept == True:
                    self.x = x_new
                    self.pred = pred_new
                    if pred_new > bestscore:
                        bestscore = pred_new
                        xBest = x_new
                        totalImprove += 1
                        self.scale = self.scale * 0.99
                
                seqs_list.append(self.sample(x_new)[0])
                pred_list.append(pred_new[0])
            
            pBadAccept = kBadAccept / (kBadAccept + kBadRefuse)
            recordPredNow.append(np.round(self.pred, 4))
            recordPredBest.append(np.round(bestscore, 4))
            recordPBad.append(np.round(pBadAccept, 4))

            if kIter%10 == 0:
                print('i:{},t(i):{:.2f}, badAccept:{:.6f}, f(x)_best:{:.6f}'.\
                    format(kIter, tNow, pBadAccept, bestscore[0]))

            tNow = tNow * alpha
            kIter = kIter + 1

        seqs_res = list(set(seqs_list))
        pred_res = list(set(pred_list))
        seqs_res.sort(key = seqs_list.index)
        pred_res.sort(key = pred_list.index)
        
        seqs_res = seqs_res[-MaxPoolsize:]
        pred_res = pred_res[-MaxPoolsize:]
        
        write_fa(self.outdir + "/ExpIter" + ".txt", seqs_res)
        write_profile(self.outdir + "/ExpIter" + ".csv", seqs_res, pred_res)
        
        print('improve:{:d}'.format(totalImprove))
        pdf = PdfPages(self.outdir+'/compared_with_natural.pdf')
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
    
    
    def Step(self, xNow): 
        xNew = xNow.copy()
        p = np.random.randint(0, xNow.shape[0])
        xNew[p] = xNow[p] + self.scale * (self.xMax[p] - self.xMin[p])* random.normalvariate(0, 1)
        xNew[p] = max(min(xNew[p], self.xMax[p]), self.xMin[p])
        return xNew

    