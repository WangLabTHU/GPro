import os, sys
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import OneHotCategorical

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

from gpro.utils.base import open_fa, write_profile, write_fa
from gpro.generator.wgan.common import onehot2seq
from gpro.generator.wgan.wgan import Generator
# from gpro.predictor.cnn_k15.cnn_k15 import CNN_K15

from tqdm import tqdm

matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

class GradientAlgorithm():
    def __init__(self,
                 predictor,
                 generator_modelpath,
                 predictor_modelpath,
                 natural_datapath,
                 
                 sample_number = 200,
                 savepath = None,
                 z_dim = 128,
                 seq_len = 50,
                 is_rnn = False
                 ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Seqs = open_fa(natural_datapath)
        
        self.generator = Generator(seq_len=seq_len).to(self.device)
        state_dict = torch.load(generator_modelpath)
        self.generator.load_state_dict(state_dict['model'])
        self.generator.eval()
        for name, parameter in self.generator.named_parameters():
            parameter.requires_grad = False
        
        self.predictor = predictor.model.to(self.device)
        state_dict = torch.load(predictor_modelpath)
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval() # for RNN, 2023.09.27
        if is_rnn:
            self.predictor.train()
        for name, parameter in self.predictor.named_parameters():
            parameter.requires_grad = False
        
        
        self.sample_number = sample_number
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
    
    
    def forward(self, noise):
        onehot = self.generator(noise)
        yp = self.predictor(onehot)
        
        seqs = onehot2seq( onehot.permute(0, 2, 1), self.seq_len)
        pred = yp.flatten().tolist()
        return yp, seqs, pred

    def evaluate(self, seqs):
        onehot = self.seq2oh(seqs)
        onehot = torch.tensor(onehot, dtype=float)
        onehot = onehot.to(torch.float32).to(self.device).permute(0,2,1)
        outputs = self.predictor(onehot)
        pred = outputs.flatten().tolist()
        return np.array(pred)
    
    def reparameterize(self, mu, sigma):   
        eps =  torch.randn(self.sample_number, self.z_dim, device=self.device)
        noise = torch.zeros([self.sample_number, self.z_dim], device=self.device)
        for i in range(eps.shape[0]):
            for j in range(eps.shape[1]):
                noise[i,j] = mu[j] + eps[i,j] * sigma[j]
        return noise
    
    def run(self,
            learning_rate = 1e-2,
            MaxPoolsize=2000,
            MaxIter=200,
            mode = "max"):
        
        
        self.outdir = self.savepath
        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)
        
        self.mu = torch.zeros([self.z_dim], requires_grad=True, device=self.device)
        self.sigma = torch.ones([self.z_dim], requires_grad=False, device=self.device)
        
        self.noise = self.reparameterize(self.mu, self.sigma)
        self.optimizer = optim.SGD([self.mu], lr=learning_rate)
        
        seqs_list = []
        pred_list = []
        noise_start = self.noise.detach().cpu().numpy()
        
        res_array = []
        
        for Iteration in range(MaxIter):
            if self.mu.grad is not None:
                self.mu.grad.zero_()
            self.noise = self.reparameterize(self.mu, self.sigma)
            self.yp, self.seqs, self.pred = self.forward(self.noise)
            if mode == "max":
                self.yp.backward(-torch.ones_like(self.yp))
            elif mode == "min":
                self.yp.backward(torch.ones_like(self.yp))
            self.optimizer.step()
            
            res_array.append(self.pred)
            
            if Iteration % 20 == 0:
                seqs_list += self.seqs
                pred_list += self.pred 
                glob_best = np.max(self.pred)
                print('Iter = ' + str(Iteration) + ' , BestScore = ' + str(glob_best))
        
        
        res_array = np.transpose(np.asarray(res_array))
        res_dataframe = pd.DataFrame(data = res_array , columns = range(MaxIter))
        res_melt = res_dataframe.melt(value_name='ylog' , var_name='iter')
        res_melt['optimization'] = mode
        res_melt.to_csv(self.outdir + '/directed_evolution_{}_{}_result.csv'.format(mode, MaxIter), sep='\t')
        
        
        seqs_res, pred_res = self.dropRep(seqs_list, pred_list)
        seqs_res = seqs_res[-MaxPoolsize:]
        pred_res = pred_res[-MaxPoolsize:]
        
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
        
        # save z
        noise_end = self.noise.detach().cpu().numpy()
        font = {'size' : 10}
        matplotlib.rc('font', **font)
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
        fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
        
        sns.histplot(noise_start[:,0], label='original', alpha = 0.75, kde=True) # bins=30
        sns.histplot(noise_end[:,0], label='model-guided', alpha = 0.75, kde=True)
        
        ax.set_xlabel('Value', fontsize=10)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.legend(loc='upper right')
        plt.title("")
        plt.show()
        plt.savefig( self.outdir + "/z.png")
        return
    
    def dropRep(self, seqs, pred):
        
        seqs_new = []
        idx_new = []
        for i, item in enumerate(seqs):
            if item not in seqs_new:
                seqs_new.append(item)
                idx_new.append(i)
        
        seqs_new = np.array(seqs_new)
        pred_new = np.array(pred)[idx_new]
        
        return seqs_new, pred_new
    

