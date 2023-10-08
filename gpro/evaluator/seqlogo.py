import os, sys
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker

import warnings
warnings.filterwarnings("ignore")

from ..utils.utils_evaluator import read_fa, seq2onehot

def plot_seqlogos(predictor, predictor_training_datapath, predictor_modelpath, report_path, file_tag, num_seqs_to_test=1000, plot_mode="saliency"):
    
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
    sns.set_style("whitegrid")
    final_arr = [grads_over_letters_on[letter]/counts_per_positon_on[letter] for letter in alph]
    seq_len = len(final_arr[0])
    
    # new arr mode could be natural seqs in utils
    
    nn_df = pd.DataFrame(data = final_arr).T
    nn_df.columns = alph # 1st row as the column names
    nn_df.index = range(len(nn_df))
    nn_df = nn_df.fillna(0)
    
    # create Logo object
    font = {'size'   : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (np.minimum(seq_len / 4, 100), np.maximum(len(alph) / 3.5, 3)), dpi = 300)

    if 'saliency' in plot_mode:
        nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'weight')
    else:
        nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'probability')

    nn_logo = logomaker.Logo(nn_df, ax = ax, color_scheme = 'classic', stack_order = 'big_on_top', fade_below = 0.8, shade_below = 0.6)
    # https://logomaker.readthedocs.io/en/latest/implementation.html

    if 'saliency' in plot_mode:
        nn_logo.ax.set_ylabel('Weight', fontsize = 12)
    else:        
        nn_logo.ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
        nn_logo.ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'], fontsize = 12)
        nn_logo.ax.set_ylabel('Probability', fontsize = 12)

    # style using Logo methods
    nn_logo.style_spines(visible = False)
    nn_logo.style_spines(spines = ['left'], visible = True)
    nn_logo.ax.set_xticks(np.arange(0, seq_len, 10))
    nn_logo.ax.set_xticklabels([str(x) for x in np.arange(0, seq_len, 10)], fontsize = 12)
    nn_logo.ax.set_xlabel('Position', fontsize = 12)

    plt.tight_layout()
    
    plotnamefinal = report_path + 'seqlogo_' + file_tag + ".png"
    print(plot_mode + ' map saved to ' + plotnamefinal)
    plt.savefig(plotnamefinal)
    # plt.savefig(plotnamefinal.split('.png')[0] + '.svg')
    

