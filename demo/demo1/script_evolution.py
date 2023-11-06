from gpro.optimizer.evolution.sswm import SSWM
from gpro.predictor.attnbilstm.attnbilstm import AttnBilstm_language

predictor = AttnBilstm_language(length=110)
predictor_modelpath = "./checkpoints/attnbilstm/checkpoint.pth"
natural_datapath = "./datasets/Random_testdata_complex_media_seq.txt"

left_flank = ''.join(['T','G','C','A','T','T','T','T','T','T','T','C','A','C','A','T','C'])
right_flank = ''.join(['G','G','T','T','A','C','G','G','C','T','G','T','T'] )

tmp = SSWM(predictor=predictor, predictor_modelpath=predictor_modelpath,
           natural_datapath=natural_datapath, savepath="./results/SSWM")
tmp.run(MaxIter=10, flanking=[left_flank, right_flank], mode="max")

tmp = SSWM(predictor=predictor, predictor_modelpath=predictor_modelpath,
           natural_datapath=natural_datapath, savepath="./results/SSWM")
tmp.run(MaxIter=10, flanking=[left_flank, right_flank], mode="min")


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc


df_max = pd.read_csv("results/SSWM/directed_evolution_max_10_result.csv", sep="\t")
df_min = pd.read_csv("results/SSWM/directed_evolution_min_10_result.csv", sep="\t")
df_concat = pd.concat([df_max,df_min])

font = {'size' : 10}
matplotlib.rc('font', **font)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
fig, ax = plt.subplots(figsize = (6,4), dpi = 300)

ax = sns.boxplot( x="edit_distance", y="expression",  data=df_concat, hue="Selection Direction", 
                  hue_order = ["min", "max"], fliersize=0, palette = ['tab:orange', 'tab:blue'])
h,_ = ax.get_legend_handles_labels()

ax.set_xlabel('Number of mutation steps', fontsize=10)
ax.set_ylabel('Predicted expression', fontsize=10)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.legend(h, ['Minimizing', 'Maximizing'], loc="lower left", bbox_to_anchor=(0.2, 1), ncol=2) # loc='upper center'
plt.title("")
plt.show()
plt.savefig("./results/SSWM_Boxplot.png")
        
        