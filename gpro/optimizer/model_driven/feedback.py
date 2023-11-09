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

from gpro.utils.base import open_fa, write_profile, write_fa, write_seq
from gpro.generator.wgan.common import onehot2seq

class Feedback():
    def __init__(self,
                 generator,
                 predictor,
                 predictor_modelpath,
                 natural_datapath,
                 
                 sample_number=1000,
                 savepath=None,
                 ):
        
        
        # define the Generator and Predictor
        self.generator = generator
        
        self.predictor = predictor
        self.predictor_modelpath = predictor_modelpath
        
        self.natural_datapath = natural_datapath
        self.sample_number = sample_number
        self.savepath = savepath
        
        self.Seqs = open_fa(natural_datapath)
        self.Pred = self.evaluate(self.Seqs)
        self.Pred = [float(item) for item in self.Pred]
        self.Seqs_nat = self.Seqs.copy()
        self.Pred_nat = self.Pred.copy()
        # self.Exps_nat = open_fa(predictor_expression_datapath)

    def forward(self, seed):
        self.generator.sample_model_path = None
        seqs = self.generator.generate(self.generator_modelpath, self.sample_number, seed=seed)
        pred = self.predictor.predict_input(self.predictor_modelpath, seqs, mode="data")
        return seqs, pred
    
    def evaluate(self, seqs):
        pred = self.predictor.predict_input(self.predictor_modelpath, seqs, mode="data")
        return pred
    
    def plot_hist(self, plot_path):
        font = {'size' : 10}
        matplotlib.rc('font', **font)
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
        fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
        
        # data = {"natural": list(self.Pred_nat), "model-guided": list(self.Pred)}
        # data = pd.DataFrame(data)
        # sns.histplot(data, alpha = 0.75, kde=True, bins=30)
        
        vmin = min(min(self.Pred_nat), min(self.Pred))
        vmax = max(max(self.Pred_nat), max(self.Pred))
        sns.histplot(self.Pred_nat, label='natural', alpha = 0.75, kde=True, bins=30, binrange=(vmin,vmax), color="tab:blue")
        sns.histplot(self.Pred, label='model-guided', alpha = 0.75, kde=True, bins=30, binrange=(vmin,vmax), color="tab:orange")

        ax.set_xlabel('Predicts', fontsize=10)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.legend(loc='upper right')
        plt.title("")
        plt.show()
        plt.savefig(plot_path)
    
    
    def plot_mean(self, plot_path):
        data_len = len(self.mean_list)
        
        font = {'size' : 12}
        matplotlib.rc('font', **font)
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
        fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
        x = list(range(0, data_len))

        palette = ['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy']
        plt.plot(x, self.mean_list, 'o-', label = 'predicts level', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:purple', alpha = 0.8)
        plt.plot(x, self.mean_list, '-', color = 'tab:purple', linewidth = 0.7)
        plt.legend(loc="lower left", markerscale = 1) # upper

        ax.set_xlabel('Iters', fontsize=12)
        ax.set_ylabel('Mean Predictions', fontsize=12)
        plt.tick_params(length = 10)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xticks(np.arange(0, data_len, 10))
        ax.set_xticklabels([str(x) for x in np.arange(0, data_len, 10)], fontsize = 12)

        plt.tight_layout()
        plt.savefig(plot_path)
        return
    
    def unique_list(self, seq, exp):
        unique_dict = {}
        unique_seq = []
        unique_exp = []

        for item, value in zip(seq, exp):
            if item not in unique_dict:
                unique_dict[item] = value
                unique_seq.append(item)
                unique_exp.append(value)

        return unique_seq, unique_exp
    
    def run(self,
            MaxIter = 20, # 200
            MaxEpoch = 50, # 500
            mode = "wgan",
            MaxPoolsize=1000):
        
        self.outdir = self.savepath
        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)
        
        self.plot_dir = self.outdir + "/plot"
        if os.path.exists(self.plot_dir) == False:
            os.makedirs(self.plot_dir)
        
        self.traj_dir = self.outdir + "/traj"
        if os.path.exists(self.traj_dir ) == False:
            os.makedirs(self.traj_dir )
        
        self.mean_list = [np.mean(self.Pred)]
        
        for Iteration in range(MaxIter):
        
            dataset_path = self.traj_dir + "/ExpIter_" + str(Iteration) + ".txt"
            checkpoint_path = self.outdir + "/checkpoints"
            write_seq(dataset_path, self.Seqs)

            if mode == "wgan":
                # args = [self.generator.seq_len, self.generator.num_epochs, self.generator.save_epoch]
                # self.generator.__init__(length=args[0], num_epochs=args[1], save_epoch=args[2])
                self.generator_modelpath = checkpoint_path + "/wgan/checkpoints/net_G_12.pth"
            elif mode == "diffusion":
                # args = [self.generator.seq_len, self.generator.epochs, self.generator.check_every]
                # self.generator.__init__(length=args[0], epochs=args[1], check_every=args[2])
                self.generator_modelpath = checkpoint_path + "/diffusion/check/checkpoint.pt"

            self.generator.train(dataset=dataset_path, savepath=checkpoint_path)
            
            seqs_list = []
            pred_list = []

            for epoch in range(MaxEpoch):
                seqs, pred = self.forward(epoch)
                seqs_list += seqs
                pred_list += pred

            idx = sorted(range(len(self.Pred)), key=lambda k: self.Pred[k], reverse=True)
            self.Seqs = np.array(self.Seqs)[idx]
            self.Pred = np.array(self.Pred)[idx]

            idx = sorted(range(len(pred_list)), key=lambda k: pred_list[k], reverse=True)
            seqs_list = np.array(seqs_list)[idx]
            pred_list = np.array(pred_list)[idx]

            seqs_res = list(self.Seqs[0:-MaxPoolsize]) + list(seqs_list[0:MaxPoolsize])
            pred_res = list(self.Pred[0:-MaxPoolsize]) + list(pred_list[0:MaxPoolsize])
            seqs_res, pred_res = self.unique_list(seqs_res, pred_res)
            bias = len(self.Seqs) - len(seqs_res)
            seqs_res = list(seqs_res) + list(seqs_list[MaxPoolsize:MaxPoolsize+bias])
            pred_res = list(pred_res) + list(pred_list[MaxPoolsize:MaxPoolsize+bias])

            print("Optimized predicted Expression Level: {}->{}".format(np.mean(self.Pred), np.mean(pred_res)))
            self.Seqs = seqs_res
            self.Pred = pred_res
            
            plot_path = self.plot_dir + "/hist_" + str(Iteration+1) + ".png"
            self.plot_hist(plot_path)
            
            self.mean_list.append(np.mean(self.Pred))
            
        plot_path = self.outdir + "/linechart" + ".png"
        self.plot_mean(plot_path)
        
        idx = sorted(range(len(self.Pred)), key=lambda k: self.Pred[k], reverse=True)
        self.Seqs = np.array(self.Seqs)[idx]
        self.Pred = np.array(self.Pred)[idx]

        write_fa(self.outdir + "/ExpIter" + ".txt", self.Seqs)
        write_profile(self.outdir + "/ExpIter" + ".csv", self.Seqs, self.Pred)
        
        pdf = PdfPages(self.outdir + '/compared_with_natural.pdf')
        plt.figure()
        
        # if len(self.Seqs) > MaxPoolsize:
        #     self.Seqs = random.sample(self.Seqs, MaxPoolsize)
        nat_score = self.Pred_nat
        plt.boxplot([nat_score, self.Pred])
        plt.ylabel('Score')
        plt.xticks([1,2],['Natural','Optimized'])
        pdf.savefig()
        pdf.close()
        return


