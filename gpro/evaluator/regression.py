import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

from ..utils.utils_evaluator import read_fa, seq2onehot

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

def plot_regression_performance(predictor_expression_datapath, predictor_prediction_datapath, report_path, file_tag, display=False):

    
    pred_Y = read_fa( predictor_prediction_datapath )
    real_Y = read_fa( predictor_expression_datapath )
    pred_Y = [float(item) for item in pred_Y]
    real_Y = [float(item) for item in real_Y]

    pred_Y = np.reshape(pred_Y, (len(pred_Y), -1))
    real_Y = np.reshape(real_Y, (len(real_Y), -1))
    
    
    # get number of samples and number of outputs
    (n_samples, n_outputs) = np.shape(real_Y)

    # calculate difference between Predicted output and target output
    diff_Y = pred_Y - real_Y
    absDiff_Y = np.abs(diff_Y)

    # compute the absolute mean, absolute standard deviation prediction-target difference:
    ad_mean_Y = np.mean(absDiff_Y, axis=0) # Mean absolute difference
    ad_std = np.std(absDiff_Y, axis=0)     # Standard deviation of the Mean absolute difference 

    # initialize empty array for R2 calculation
    ad_r2 = np.zeros_like(ad_mean_Y)       # R2

    # create Graphs
    # R2 (Coefficient of Determination)
    index = 0 # because just one predictor
    ad_r2[index] = r2(pred_Y[:,index], real_Y[:,index])

    import matplotlib
    from matplotlib import rc
    font = {'size'   : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        ## for Palatino and other serif fonts use:
        #rc('font',**{'family':'serif','serif':['Palatino']})
        #rc('text', usetex=True)

    # change font
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    sns.set_style("whitegrid")

    # Display Output Values
    x_tot=np.squeeze(pred_Y[:,index])
    y_tot=np.squeeze(real_Y[:,index])
    pearson = stats.pearsonr(x_tot, y_tot)[0]
    spearman = stats.spearmanr(x_tot, y_tot)[0]
    
    # NOTE: for sklearn r2 need to have y_true, y_pred order 
    #g = sns.jointplot(x_tot, y_tot, kind="reg", color="b", stat_func=self.r2) # stat_func argument deprecated
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)
    g = sns.jointplot(x=x_tot, y=y_tot, kind="reg", color="cornflowerblue")
    g.ax_joint.text = r2
    g.plot_joint(plt.scatter, c="cornflowerblue", s=4, linewidth=1, marker=".", alpha=0.2) # 0.08
    try:
    	g.plot_joint(sns.kdeplot, zorder=0, color="grey", n_levels=6) # shade=False
    except: # if discrete, then need hist instead of kde
    	g.plot_joint(sns.histplot, zorder=0, color="grey",alpha=0.05)
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("Predicted", "Experimental");

    # save the figure
    g.savefig(report_path +'regression_' + file_tag + ".png", bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    ad_pearson = pearson
    ad_spearman = spearman
        
    # store model performance metrics for return   
    deploy_test_metrics = [ad_mean_Y[index], ad_std[index], ad_r2[index], ad_pearson, ad_spearman]
        
    return deploy_test_metrics