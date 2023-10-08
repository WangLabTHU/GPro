import os, sys
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from tqdm import tqdm

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from ..utils.utils_evaluator import read_fa, seq2onehot


def plot_saliency_map(predictor, predictor_training_datapath, predictor_modelpath, report_path, file_tag, num_seqs_to_test=100):
    
    # loading models
    device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),] 
    model = predictor.model.to(device)
    state_dict = torch.load(predictor_modelpath)
    model.load_state_dict(state_dict)
    
    # freezing parameters
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    
    seqs = read_fa(predictor_training_datapath)
    seq_len = len(seqs[0])
    alph = ["T", "C", "G", "A"]
    rand_indices = np.random.choice(range(len(seqs)), num_seqs_to_test, replace=False)
    
    # look at average saliency based on original nucleotide
    grads_over_letters_on = {c: np.zeros(seq_len) for c in alph}
    counts_per_positon_on = {c: np.zeros(seq_len) for c in alph}
    
    seqs_onehot = seq2onehot(seqs, seq_len)
    seqs_onehot = torch.tensor(seqs_onehot, dtype=float)
    seqs_onehot = seqs_onehot.permute(0,2,1)

    for idx in tqdm(rand_indices): 
        onehot = seqs_onehot[idx,:,:].to(device).to(torch.float32).unsqueeze(0).requires_grad_(True)
        outputs = torch.mean(model(onehot))
        outputs.backward()
        grads = onehot.grad.permute(0,2,1)[0]
        grads = grads.cpu().numpy()
        
        for pos, grad in enumerate(grads): 
            nt = seqs[idx][pos]
            # grad_at_pos = grad[alph.index(nt)]
            grad_at_pos = np.abs(grad[alph.index(nt)])
            grads_over_letters_on[nt][pos] += grad_at_pos
            counts_per_positon_on[nt][pos] += 1
                
    cmap = sns.color_palette("PuBu", n_colors = 20)
    final_arr = [grads_over_letters_on[letter]/counts_per_positon_on[letter] for letter in alph]
    
    font = {'size'   : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (np.minimum(seq_len / 4, 100),np.maximum(len(alph) / 3.5, 3)), dpi = 300)
    plt.rcParams["figure.dpi"] = 300
    
    g = sns.heatmap(final_arr, cmap=cmap,  cbar_kws={"orientation": "vertical", "pad": 0.035, "fraction": 0.05})
    ax.tick_params(length = 10)
    plt.xticks(np.arange(0, seq_len, 10), np.arange(0, seq_len, 10), fontsize = 15, rotation = 0)
    plt.yticks(np.arange(0.5, len(alph)), np.arange(0.5, len(alph)), fontsize = 15, rotation = 0)

    g.set_yticklabels(alph, fontsize = 20, rotation = 0)        
    g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
    plt.xlabel('Position', fontsize = 20)

    sal_or_act = "saliency"
    
    cbar = ax.collections[0].colorbar
    cbar.set_label(sal_or_act + "  ", rotation = 270, fontsize = 20)
    ax.collections[0].colorbar.ax.set_yticklabels(ax.collections[0].colorbar.ax.get_yticklabels())
    cbar.ax.get_yaxis().labelpad = 30
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length = 10, labelsize=15)

    plotnamefinal = report_path + 'saliency_' + file_tag + ".png"
    print(sal_or_act + ' map saved to ' + plotnamefinal)
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    return(final_arr, plotnamefinal, alph)
    

def plot_activation_map(predictor, predictor_training_datapath, predictor_modelpath, report_path, file_tag, num_seqs_to_test=20):
    # loading models
    device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),] 
    model = predictor.model.to(device)
    state_dict = torch.load(predictor_modelpath)
    model.load_state_dict(state_dict)
    
    # freezing parameters
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    
    seqs = read_fa(predictor_training_datapath)
    seq_len = len(seqs[0])
    seqs_onehot = seq2onehot(seqs, seq_len)
    seqs_onehot = torch.tensor(seqs_onehot, dtype=float)
    seqs_onehot = seqs_onehot.permute(0,2,1)
    
    max_values_list = []
    activation_list = []
    
    for i in tqdm(range(len(seqs_onehot))):
        onehot = seqs_onehot[i,:,:].to(device).to(torch.float32).unsqueeze(0).requires_grad_(True)
        outputs = torch.mean(model(onehot))
        outputs.backward()
        grads = onehot.grad
        grads_abs = (torch.abs(grads)).cpu().numpy()
        grads_max = (np.max(grads_abs, axis=1)).ravel()
        
        max_values_list.append( max(grads_max) )
        activation_list.append( grads_max )

    idx = sorted(range(len(max_values_list)), key=lambda k: max_values_list[k], reverse=True)
    max_values_list = (np.array(max_values_list))[idx][0:num_seqs_to_test]
    activation_list = (np.array(activation_list))[idx][0:num_seqs_to_test]
    seqs_activation = (np.array(seqs))[idx][0:num_seqs_to_test]
    
    final_arr = [activation_list[i]/max_values_list[i] for i in range(len(activation_list))]
    cmap = sns.color_palette("PuBu", n_colors = 20)
    
    font = {'size'   : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (np.minimum(seq_len / 4, 100),np.maximum(num_seqs_to_test / 3.5, 3)), dpi = 300)
    plt.rcParams["figure.dpi"] = 300
    
    g = sns.heatmap(final_arr, cmap=cmap, cbar_kws={"orientation": "vertical", "pad": 0.035, "fraction": 0.05})
    ax.tick_params(length = 10)
    plt.xticks(np.arange(0, seq_len, 10), np.arange(0, seq_len, 10), fontsize = 15, rotation = 0)
    plt.yticks(np.arange(0.5, num_seqs_to_test), np.arange(0.5, num_seqs_to_test), fontsize = 15, rotation = 0)

    g.set_yticklabels(range(num_seqs_to_test), fontsize = 20, rotation = 0)        
    g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
    plt.xlabel('Position', fontsize = 20)
    
    sal_or_act = "activation"
    
    cbar = ax.collections[0].colorbar
    cbar.set_label(sal_or_act + "  ", rotation = 270, fontsize = 20)
    ax.collections[0].colorbar.ax.set_yticklabels(ax.collections[0].colorbar.ax.get_yticklabels())
    cbar.ax.get_yaxis().labelpad = 30
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length = 10, labelsize=15)

    plotnamefinal = report_path + 'activation_' + file_tag + ".png"
    print(sal_or_act + ' map saved to ' + plotnamefinal)
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    # plt.savefig(plotnamefinal.split('.png')[0] + '.svg')
    return(seqs_activation, plotnamefinal)