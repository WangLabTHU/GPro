import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from gpro.predictor.attnbilstm.attnbilstm import TorchStandardScaler, AttnBilstm_language
from gpro.utils.base import open_fa

######################
###   Preparation  ###
######################

scaler_save_path = "./checkpoints/scaler_params.npy"
scaler_paras = np.load(scaler_save_path, allow_pickle=True).item()
scaler = TorchStandardScaler(mean = scaler_paras["mean"], std = scaler_paras["std"])
print("Scaler Params: ", scaler.getvalue())


######################
####   Training   ####
######################


model = AttnBilstm_language(length=110, epoch=200, exp_mode="direct", patience = 200)

dataset = "./datasets/tmp_seq.txt"
labels = './datasets/tmp_exp.txt'

save_path = './checkpoints/'
model.train(dataset=dataset,labels=labels,savepath=save_path)


###########################
##### Predicting Mode #####
###########################


model_path = "./checkpoints/attnbilstm/checkpoint.pth"
data_path = "./datasets/test_seq.txt"
test_seqs =  open_fa("./datasets/test_seq.txt")
test_label = open_fa("./datasets/test_exp.txt")
test_label = [float(item) for item in test_label]

predict = []
test_seqs_chunk = [test_seqs[i:i + 200] for i in range(0, len(test_seqs), 200)]

for i in range(len(test_seqs_chunk)):
    output = model.predict_input(model_path=model_path, inputs=test_seqs_chunk[i], mode="data")
    predict += output



predict = np.array(predict)
predict = scaler.inv_transform(predict.reshape(1, -1)).reshape(predict.shape)
print(test_label[0:5])
print(predict[0:5])

cor_pearsonr = pearsonr(test_label, predict)
print(cor_pearsonr)

###########################
#####   Displaying    #####
###########################


path_csv = "./results/Pytorch_test_tpu_model.csv"
res_dict = {"seqs": test_seqs, "real": test_label, "pred":predict}
df = pd.DataFrame(res_dict)
df.to_csv(path_csv)

import scipy
import seaborn as sns
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update(matplotlib.rcParamsDefault)
rcParams['pdf.fonttype'] = 42

sns.set(context = 'paper' , style='ticks' , 
        rc = {
            'figure.autolayout' : True,
            'axes.titlesize' : 8 ,
            'axes.titleweight' :'bold',
            
            'figure.titleweight' : 'bold' ,
            'figure.titlesize' : 8 ,
            
            'axes.labelsize' : 8 ,
            'axes.labelpad' : 2 ,
            'axes.labelweight' : 'bold' , 
            'axes.spines.top' : False,
            'axes.spines.right' : False,
            
            'xtick.labelsize' : 7 ,
            'ytick.labelsize' : 7 ,
            
            'legend.fontsize' : 7 ,
            'figure.figsize' : (3.5, 3.5/1.6 ) ,          
            
            'xtick.direction' : 'out' ,
            'ytick.direction' : 'out' ,
            
            'xtick.major.size' : 2 ,
            'ytick.major.size' : 2 ,
            
            'xtick.major.pad' : 2,
            'ytick.major.pad' : 2,
            
            #'lines.linewidth' : 1            
            
             } )

tpu_results_df = pd.read_csv('./results/Pytorch_test_tpu_model.csv', sep =',' , index_col = 0)
reg = 1
fig_name = 'Pytorch_test_complex'
fig_file = fig_name+"_performance"
fig=plt.figure(figsize=(9,9) , dpi= 300, facecolor='w', edgecolor='k')
fig.tight_layout(pad = 1)

x = list(tpu_results_df['pred'].values)
y = list(tpu_results_df['real'].values)

xlabel = 'TPU Predicted Expression'
ylabel = 'Measured Expression'


r = scipy.stats.pearsonr(x ,y )
if reg: 
    sns.regplot(x=x ,y=y ,
                scatter_kws= {'s':1,'linewidth':0, 'rasterized':True} ,
                line_kws= {'linewidth':2} ,
               color= '#0868ac', robust = 1 )
else : 
    sns.scatterplot(x=x ,y=y ,s=1,linewidth=0, rasterized=True , color= '#0868ac')

ax = plt.gca()


ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
if (r[1] ==0.0) :
    ax.set_title(f"PCC = {r[0] : 0.3f} | P < {np.nextafter(0, 1) : 0.0E} | N = {len(x)}"  )
else :
    ax.set_title(f"PCC = {r[0] : 0.3f} | P = {r[1] : 0.2E} | N = {len(x)}"  )


plt.setp(ax.artists, edgecolor = 'k')
plt.setp(ax.lines, color='k')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)

plt.savefig("./results/%s.pdf" % (fig_file,), bbox_inches="tight")
plt.savefig("./results/%s.png" % (fig_file,), bbox_inches="tight")

plt.show()
