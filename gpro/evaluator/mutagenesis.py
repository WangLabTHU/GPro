import os, sys
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from scipy import stats
from scipy import mean
from scipy.stats import sem, t

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from ..utils.utils_evaluator import read_fa, seq2onehot

def get_new_mismatch_seqs(listofseqs):
    mismatches = []
    alph = ["T", "C", "G", "A"]
    
    for origin_seq in listofseqs:
        origin_seq_list = list(origin_seq)
        for position in range(len(origin_seq)):
            mismatch_seq_list = origin_seq_list.copy()
            for nt in alph:
                mismatch_seq_list[position] = nt
                mismatches.append("".join(mismatch_seq_list))
    return mismatches

def get_std_dev_at_each_bp(mismatches, num_seqs_to_test, seq_len):
    alph = ["T", "C", "G", "A"]
    alph_len = len(alph)
    all_val_std_devs = np.zeros((num_seqs_to_test, seq_len))
    curr_index_of_seqs = 0

    for row in range(0, num_seqs_to_test):
        for col in range(0, seq_len):
            val_for_current_seqs = mismatches[curr_index_of_seqs:(curr_index_of_seqs+alph_len)]
            val_std_dev = np.std(list(val_for_current_seqs))   
            all_val_std_devs[row, col] = val_std_dev           
            curr_index_of_seqs = curr_index_of_seqs + alph_len    
    return all_val_std_devs
    

def get_matrix(listofseqs, num_seqs_to_test, predictor, predictor_modelpath):
    alph = ["T", "C", "G", "A"]
    if num_seqs_to_test > len(listofseqs):
        num_seqs_to_test = len(listofseqs)
    
    listofseqs = list(listofseqs)
    listofseqs = random.sample(listofseqs, num_seqs_to_test)
    mismatches = get_new_mismatch_seqs(listofseqs)
    seq_len = len(listofseqs[0])
    
    device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),] 
    model = predictor.model.to(device)
    state_dict = torch.load(predictor_modelpath)
    model.load_state_dict(state_dict)
    model.eval()
    
    mismatches_onehot = seq2onehot(mismatches, seq_len)
    mismatches_onehot = torch.tensor(mismatches_onehot, dtype=float)
    mismatches_onehot = mismatches_onehot.permute(0,2,1)
    mismatches_preds = []
    
    for i in tqdm(range(len(mismatches_onehot))):
        onehot = mismatches_onehot[i,:,:].to(device).to(torch.float32).unsqueeze(0)
        outputs = model(onehot)
        mismatches_preds.append(outputs.tolist()[0])
    
    matrix_of_std = get_std_dev_at_each_bp(mismatches_preds, num_seqs_to_test, seq_len)
    return matrix_of_std
    
def get_means_and_bounds_of_stds(matrix):
    """get the means and conf intervals for std deviations in order to plot
    Parameters
    ----------
    matrix : list of numpy array of size (num_seqs_to_test, seq_len) with all std deviations of predictions
    
    Returns
    -------
    means : list of means of std deviations of predictions across the length of the sequence
    conf_int_lower_bound : list of lower confidence intervals for plotting
    conf_int_upper_bound : list of upper confidence intervals for plotting
    """ 
    
    means = []
    conf_int_lower_bound = []
    conf_int_upper_bound = []
    
    matrix = np.array(matrix)

    for column in matrix.T:
        confidence = 0.95 # compute 95% confidence interval
        n = len(column) # sequence num
        m = mean(column) # average std over position
        std_err = sem(column)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        means.append(m)
        conf_int_lower_bound.append(m-h)
        conf_int_upper_bound.append(m+h)

    return([means, conf_int_lower_bound, conf_int_upper_bound])



'''
plot_kmer_with_model(generator, generator_modelpath, generator_training_datapath,  report_path, file_tag, K=6, num_seqs_to_test=10000)
plot_kmer(generator_training_datapath, generator_sampling_datapath, report_path, file_tag, K=6, num_seqs_to_test=10000)

plot_mutagenesis(seqs, exps, predictor, predictor_modelpath, report_path, file_tag, num_seqs_to_test=200)
plot_mutagenesis(predictor, predictor_modelpath, predictor_training_datapath, predictor_expression_datapath, report_path, file_tag, num_seqs_to_test=200)
'''

def plot_mutagenesis(predictor, predictor_modelpath, predictor_training_datapath, predictor_expression_datapath, report_path, file_tag, num_seqs_to_test=200):
    
    seqs = read_fa(predictor_training_datapath)
    exps = read_fa(predictor_expression_datapath)
    exps = [float(item) for item in exps]
    
    real_X = np.array(seqs)
    real_Y = np.array(exps)
    seq_len = len(seqs[0])
    
    average_matrix_of_stds = []
    best_matrix_of_stds = []
    worst_matrix_of_stds = []

    best =  real_X[real_Y >= np.quantile(real_Y, .9)]
    worst = real_X[real_Y <  np.quantile(real_Y, .1)]
    average = real_X[[(x < np.quantile(real_Y, .8) and x > np.quantile(real_Y, 0.2)) for x in real_Y]]
    
    best_matrix_of_stds = get_matrix(best, num_seqs_to_test, predictor, predictor_modelpath)
    worst_matrix_of_stds = get_matrix(worst, num_seqs_to_test, predictor, predictor_modelpath)
    average_matrix_of_stds = get_matrix(average, num_seqs_to_test, predictor, predictor_modelpath)

    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
    x = list(range(0, seq_len))
    palette = ['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy']

    # plot random
    if len(average_matrix_of_stds) > 0:
        means, conf_int_lower_bound, conf_int_upper_bound = get_means_and_bounds_of_stds(average_matrix_of_stds)
        plt.plot(x, means, 'v-', color='grey', label = 'Random', linewidth = 0.7, markeredgecolor = 'black',markersize=5, alpha = 0.8)
        plt.plot(x, means, '-', color = 'grey', linewidth = 0.7)
        plt.plot(x, conf_int_lower_bound, '--', color='grey', linewidth=0.5)
        plt.plot(x, conf_int_upper_bound, '--', color='grey', linewidth=0.5)
        ax.fill_between(x, conf_int_lower_bound, conf_int_upper_bound, color = 'grey', alpha = 0.2)
    
    #plot worst
    if len(worst_matrix_of_stds) > 0:
        means, conf_int_lower_bound, conf_int_upper_bound = get_means_and_bounds_of_stds(worst_matrix_of_stds)
        plt.plot(x, means, '^-', color = 'tab:orange', linewidth = 0.7, label = 'Bottom 10%',markeredgecolor = 'black', markersize=5, alpha = 0.8)
        plt.plot(x, means, '-', color = 'tab:orange', linewidth = 0.7)
        plt.plot(x, conf_int_lower_bound, '--', linewidth=0.5, color = 'tab:orange')
        plt.plot(x, conf_int_upper_bound, '--', linewidth=0.5, color = 'tab:orange')
        ax.fill_between(x, conf_int_lower_bound, conf_int_upper_bound, color = 'tab:orange', alpha = 0.2)

    # plot best
    if len(best_matrix_of_stds) > 0:
        means, conf_int_lower_bound, conf_int_upper_bound = get_means_and_bounds_of_stds(best_matrix_of_stds)
        plt.plot(x, means, 'o-', label = 'Top 10%', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:blue', alpha = 0.8)
        plt.plot(x, means, '-', color = 'tab:blue', linewidth = 0.7)
        plt.plot(x, conf_int_lower_bound, '--', linewidth=0.5,  color = 'tab:blue')
        plt.plot(x, conf_int_upper_bound, '--', linewidth=0.5,  color = 'tab:blue')
        ax.fill_between(x, conf_int_lower_bound, conf_int_upper_bound, color = 'tab:blue', alpha = 0.2)

    plt.legend(loc="upper left", markerscale = 1)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel("Std Dev of Subunit Mismatch", fontsize=12)
    plt.tick_params(length = 10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(0, seq_len, 10))
    ax.set_xticklabels([str(x) for x in np.arange(0, seq_len, 10)], fontsize = 12)
    
    plotnamefinal = report_path + 'mutagenesis_' + file_tag + ".png"
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    # plt.savefig(plotnamefinal.split('.png')[0] + '.svg')

    print('In silico mutagenesis plot saved to ' + plotnamefinal)
    return (average_matrix_of_stds, plotnamefinal)


