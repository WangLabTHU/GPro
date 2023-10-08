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

import copy
from tqdm import tqdm

class Drift():
    def __init__(self,
                 predictor,
                 predictor_modelpath,
                 natural_datapath,
                 savepath=None,
                 ):
        
        
        # define the Generator and Predictor
        
        self.predictor = predictor
        self.predictor_modelpath = predictor_modelpath
        
        self.natural_datapath = natural_datapath
        self.savepath = savepath
        
        self.Seqs = open_fa(natural_datapath)
        self.Pred = self.evaluate(self.Seqs)
        self.Pred = [float(item) for item in self.Pred]
        self.Seqs_nat = self.Seqs.copy()
        self.Pred_nat = self.Pred.copy()

        self.seq_len = len(self.Seqs[0])
        
    
    def evaluate(self, seqs):
        pred = self.predictor.predict_input(self.predictor_modelpath, seqs, mode="data")
        return pred
    
    def population_remove_flank(self, population) : 
        lb = len(self.flanking[0])
        ub = len(self.flanking[1])
        
        return_population = []
        for i in range(len(population)): 
            return_population= return_population + [(population[i][lb:-ub])]
        return return_population

    def population_add_flank(self, population) : 
        left_flank = self.flanking[0]
        right_flank = self.flanking[1]
        population = copy.deepcopy(population)
        for ind in range(len(population)) :
            if not population[ind]!=population[ind]:   
                population[ind] =  left_flank+ ''.join(population[ind]) + right_flank
            else :
                print(ind)
        return population
    
    def population_mutator(self, population_current) :
        if (self.flanking!=None):
            population_current = self.population_remove_flank(population_current)
        population_next = []  
         
        for i in range(len(population_current)) :         
            for j in range(self.seq_len) : 
                population_next.append(list(population_current[i]))
                population_next.append(list(population_current[i]))
                population_next.append(list(population_current[i]))

                if (population_current[i][j] == 'A') :
                    population_next[3*(self.seq_len*i + j) ][j] = 'C'
                    population_next[3*(self.seq_len*i + j) + 1][j] = 'G'
                    population_next[3*(self.seq_len*i + j) + 2][j] = 'T'

                elif (population_current[i][j] == 'C') :
                    population_next[3*(self.seq_len*i + j)][j] = 'A'
                    population_next[3*(self.seq_len*i + j) + 1][j] = 'G'
                    population_next[3*(self.seq_len*i + j) + 2][j] = 'T'

                elif (population_current[i][j] == 'G') :
                    population_next[3*(self.seq_len*i + j)][j] = 'C'
                    population_next[3*(self.seq_len*i + j) + 1][j] = 'A'
                    population_next[3*(self.seq_len*i + j) + 2][j] = 'T'

                elif (population_current[i][j] == 'T') :
                    population_next[3*(self.seq_len*i + j)][j] = 'C'
                    population_next[3*(self.seq_len*i + j) + 1][j] = 'G'
                    population_next[3*(self.seq_len*i + j) + 2][j] = 'A'
        if (self.flanking!=None):
            population_next = self.population_add_flank(population_next) 
        return list(population_next)
    
    
    def neutral_next_generation(self, population_current): 
        population_next_all = self.population_mutator(list(population_current))
        population_next_neutral_seq = list(population_current)  
        
        for i in tqdm(range(len(population_current))) : 
            j = np.random.choice(3*self.seq_len)    
            population_next_neutral_seq[i] = population_next_all[3*self.seq_len*i + j]

        population_next_neutral_seq_fitness = self.evaluate(list(population_next_neutral_seq))
        return list(population_next_neutral_seq) , list(population_next_neutral_seq_fitness)
    
    def run(self,
            MaxIter = 5,
            flanking=None):
        
        self.outdir = self.savepath
        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)
        self.flanking = flanking
        
        if (self.flanking):
            self.seq_len = len(self.Seqs[0]) - len(self.flanking[0]) - len(self.flanking[1])
            print("Flanking Left: ", self.flanking[0])
            print("Flanking Right: ", self.flanking[1])
        
        self.mean_list = [np.mean(self.Pred)]
        
        seqs_list = [self.Seqs_nat]
        pred_list = [self.Pred_nat]
        seqs_res = [] + self.Seqs_nat
        pred_res = [] + self.Pred_nat
        for Iteration in range(MaxIter):
            generation_neutral_seq, generation_neutral_seq_fitness = self.neutral_next_generation(self.Seqs)
            generation_neutral_seq = [''.join(item) for item in generation_neutral_seq]
            
            self.Seqs = generation_neutral_seq
            self.Pred = generation_neutral_seq_fitness
            seqs_res += generation_neutral_seq
            pred_res += generation_neutral_seq_fitness
            seqs_list.append(generation_neutral_seq)
            pred_list.append(generation_neutral_seq_fitness)   
        
        res_array = np.transpose(np.asarray(pred_list))
        res_dataframe = pd.DataFrame(data = res_array , index = seqs_list[0])
        res_melt = res_dataframe.melt(value_name='expression' , var_name='edit_distance')
        res_melt.to_csv(self.outdir + '/random_walking_result.csv', sep='\t')
        
        
        idx = sorted(range(len(pred_res)), key=lambda k: pred_res[k], reverse=True)
        seqs_res = np.array(seqs_res)[idx][0:len(self.Seqs_nat)]
        pred_res = np.array(pred_res)[idx][0:len(self.Seqs_nat)]
        
        write_fa(self.outdir + "/ExpIter" + ".txt", seqs_res)
        write_profile(self.outdir + "/ExpIter" + ".csv", seqs_res, pred_res)
        
        pdf = PdfPages(self.outdir + '/compared_with_natural.pdf')
        plt.figure()
        nat_score = self.Pred_nat
        plt.boxplot([nat_score, pred_res])
        plt.ylabel('Score')
        plt.xticks([1,2],['Natural','Optimized'])
        pdf.savefig()
        pdf.close()
        
        pdf = PdfPages(self.outdir+'/each_iter_distribution.pdf')
        plt.figure()
        plt.boxplot(pred_list[1:])
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        pdf.savefig()
        pdf.close()
        
        return

