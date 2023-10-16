from gpro.generator.wgan.wgan import WGAN_language
from gpro.evaluator.kmer import plot_kmer_with_model, plot_kmer


######################
####   Generator   ###
######################

dataset_path = './datasets/train_seq.txt'
checkpoint_path = './checkpoints'
model = WGAN_language(length=1000, num_epochs=12, print_epoch = 1, save_epoch = 12)
model.train(dataset=dataset_path, savepath=checkpoint_path)

generator_modelpath = "./checkpoints/wgan/checkpoints/net_G_12.pth"
generator_sampling_datapath = "./checkpoints/wgan/samples/sample_ep12_s0_num_3000.txt"
model.generate(generator_modelpath, 3000)

plot_kmer(dataset_path, generator_sampling_datapath, report_path="./results/", file_tag="wgan")

######################
####   Predictor   ###
######################

from gpro.predictor.attnbilstm.attnbilstm import AttnBilstm_language
from gpro.evaluator.regression import plot_regression_performance

model = AttnBilstm_language(length=1000, epoch=200, exp_mode="direct", patience = 200)

dataset = './datasets/train_seq.txt'
labels = './datasets/train_exp.txt'
save_path = './checkpoints/'
model.train(dataset=dataset,labels=labels,savepath=save_path)

predictor_modelpath = "./checkpoints/attnbilstm/checkpoint.pth"
model.predict(model_path=predictor_modelpath, data_path="./datasets/test_seq.txt")

predictor_expression_datapath = "./datasets/test_exp.txt"
predictor_prediction_datapath = "./checkpoints/attnbilstm/preds.txt"
metrics = plot_regression_performance( predictor_expression_datapath, predictor_prediction_datapath,
                                          report_path="./results/", file_tag="AttnBiLSTM")
print("ad_mean_Y: {}, ad_std:{}, ad_r2:{}, ad_pearson:{}, ad_spearman:{} \n".format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))


######################
####   Optimizer   ###
######################

from gpro.optimizer.model_driven.gradient import GradientAlgorithm

predictor = AttnBilstm_language(length=1000)
generator_modelpath = "./checkpoints/wgan/checkpoints/net_G_12.pth"
predictor_modelpath = "./checkpoints/attnbilstm/checkpoint.pth"
natural_datapath = "./datasets/test_seq.txt"

df_concat = GradientAlgorithm( predictor = predictor, sample_number=100, seq_len=1000, is_rnn=True,
                               generator_modelpath=generator_modelpath, predictor_modelpath=predictor_modelpath,
                               natural_datapath=natural_datapath, savepath="./results/Gradient")
df_concat.run(mode="max", learning_rate=0.1, MaxIter=1000)
df_concat.run(mode="min", learning_rate=0.1, MaxIter=1000)


######################
####   Displaying  ###
######################

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from matplotlib.pylab import mpl


df_max = pd.read_csv("results/Gradient/directed_evolution_max_1000_result.csv", sep="\t")
df_min = pd.read_csv("results/Gradient/directed_evolution_min_1000_result.csv", sep="\t")
df_concat = pd.concat([df_min,df_max])
df_concat['ylog'] = np.log2(df_concat['ylog'])


font = {'size' : 10}
matplotlib.rc('font', **font)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
fig, ax = plt.subplots(figsize = (10,6), dpi = 1000)


df_concat.iter.sort_values().unique()
ax = sns.scatterplot(data=df_concat,
               x='iter',y='ylog',
               hue='optimization',alpha=0.4, hue_order = ["min", "max"],
               palette=['steelblue','indianred'], linewidth=0.2)
avg = (df_concat
         .groupby(['optimization','iter'])
         .ylog.median()
         .reset_index()
      )
plt.plot(avg.query('optimization=="max"').iter,
         avg.query('optimization=="max"').ylog,
         c='k')
plt.plot(avg.query('optimization=="min"').iter,
         avg.query('optimization=="min"').ylog,
         c='k',label='avg.')

plt.ylim(-3,6)
plt.xlabel('Optimizer iteration')
plt.ylabel('Predicted gene expression per\ngenerated sequence variant, log TPM')

handles, labels = ax.get_legend_handles_labels()

plt.legend([handles[0],handles[2],handles[1]],
           ['min.','avg.','max.'],
           bbox_to_anchor=(1,1)
          )
plt.savefig("./results/Demo.png")
