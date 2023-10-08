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

'''
just for wgan!!!
'''

class GeneticAlgorithm():
    def __init__(self,
                 generator_modelpath,
                 predictor_modelpath,
                 natural_datapath,
                
                 sample_number=200,       
                 savepath = None,
                 z_dim = 128,
                 seq_len = 50
                 ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Seqs = open_fa(natural_datapath)
        
        self.generator = Generator(seq_len=seq_len).to(self.device)
        state_dict = torch.load(generator_modelpath)
        self.generator.load_state_dict(state_dict['model'])
        self.generator.eval()
        
        self.predictor = CNN_K15(input_length=seq_len).to(self.device)
        state_dict = torch.load(predictor_modelpath)
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        
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
            P_rep=0.3,
            P_new=0.25,
            P_elite=0.25,
            MaxIter=1000):
        
        self.outdir = self.savepath
        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)
        
        self.MaxPoolsize = MaxPoolsize
        self.P_rep = P_rep
        self.P_new = P_new
        self.P_elite = P_elite
        self.MaxIter = MaxIter
        
        self.x = np.random.normal(size=(self.sample_number, self.z_dim))
        self.pred = self.evaluate(self.sample(self.x))
        self.oh = self.seq2oh(self.sample(self.x))
        
        idx = sorted(range(len(self.pred)), key=lambda k: self.pred[k], reverse=True)
        self.x = self.x[idx]
        self.pred = self.pred[idx]
        self.oh = self.oh[idx]
        
        bestscore = self.pred[0]
        scores = []
        
        seqs_list = []
        
        ## Genetic Algorithm
        for iteration in range(1,1+self.MaxIter):
            Poolsize = self.pred.shape[0]
            Nnew = math.ceil(Poolsize*self.P_new)
            Nelite = math.ceil(Poolsize*self.P_elite)
            IParent = self.select_parent( Nnew, Nelite, Poolsize)
            Parent = self.x[IParent,:].copy()
            
            x_new = self.act(Parent)
            self.x = np.concatenate([self.x, x_new])
            pred_new = self.evaluate(self.sample(x_new))
            self.pred = np.append(self.pred,pred_new)
            oh_new = self.seq2oh(self.sample(x_new))
            self.oh = np.concatenate([self.oh, oh_new])
            
            idx = sorted(range(len(self.pred)), key=lambda k: self.pred[k], reverse=True)
            self.x = self.x[idx]
            self.pred = self.pred[idx]
            self.oh = self.oh[idx]

            I = self.delRep(self.oh ,P_rep)
            self.x = np.delete(self.x,I,axis=0)
            self.pred = np.delete(self.pred,I,axis=0)
            self.oh  = np.delete(self.oh ,I,axis=0)
            
            self.x = self.x[:MaxPoolsize, :]
            self.pred = self.pred[:MaxPoolsize]
            self.oh = self.oh[:MaxPoolsize, :, :]
            
            
            print('Iter = ' + str(iteration) + ' , BestScore = ' + str(self.pred[0]))
            if iteration%100 == 0:
                seqs = self.sample(self.x)
                pred = self.pred
                write_profile(self.outdir + "/ExpIter" + str(iteration) + ".csv", seqs, pred)
                print('Iter {} was saved!'.format(iteration))

                seqs_list.append(seqs)
                scores.append(self.pred)
                if np.max(scores[-1]) > bestscore:
                    bestscore = np.max(scores[-1])
                else:
                    break
        
        seqs_res = (seqs_list[-1])[-MaxPoolsize:]
        write_fa(self.outdir + "/ExpIter" + ".txt", seqs_res)
        
        pdf = PdfPages(self.outdir+'/each_iter_distribution.pdf')
        plt.figure()
        plt.boxplot(scores)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        pdf.savefig()
        pdf.close()
        
        pdf = PdfPages(self.outdir+'/compared_with_natural.pdf')
        plt.figure()
        
        if len(self.Seqs) > MaxPoolsize:
            self.Seqs = random.sample(self.Seqs, MaxPoolsize)
        nat_score = self.evaluate(self.Seqs)
        plt.boxplot([nat_score,scores[-1]])
        plt.ylabel('Score')
        plt.xticks([1,2],['Natural','Optimized'])
        pdf.savefig()
        pdf.close()
        return
    
    def select_parent(self, Nnew, Nelite, Poolsize):
        ParentFromElite  = min(Nelite, Nnew//2)
        ParentFromNormal = min(Poolsize - Nelite, Nnew - ParentFromElite)
        I_e = random.sample([ i for i in range(Nelite)], ParentFromElite)
        I_n = random.sample([ i + Nelite for i in range(Poolsize - Nelite)], ParentFromNormal)
        I = I_e + I_n
        return I
    
    ## Single Mutation
    def PMutate(self, z): 
        p = np.random.randint(0,z.shape[0])
        z[p] = np.random.normal()
        return
    
    ## Random recombination
    def Reorganize(self, z, Parent):
        index = np.random.randint(0, 1,size=(z.shape[0]))
        j = np.random.randint(0, Parent.shape[0])
        for i in range(z.shape[0]):
            if index[i] == 1:
                z[i] = Parent[j,i].copy()
        return
    
    def act(self, Parent):
        for i in range(Parent.shape[0]):
            action = np.random.randint(0,1)
            if action == 0:
                self.PMutate(Parent[i,:])
            elif action == 1:
                self.Reorganize(Parent[i,:], Parent)
        return Parent
    
    def delRep(self, Onehot, p):
        I = set()
        n = Onehot.shape[0]
        i = 0
        while i < n-1:
            if i not in I:
                a = Onehot[i,:,:]
                a = np.reshape(a,((1,)+a.shape))
                a = np.repeat(a,n-i-1,axis=0)
                I_new = np.where(( np.sum(np.abs(a - Onehot[(i+1):,:,:]),axis=(1,2)) / (Onehot.shape[1]*2) ) < p)[0]
                I_new = I_new + i+1
                I = I|set(I_new)
            i += 1
        return list(I)
