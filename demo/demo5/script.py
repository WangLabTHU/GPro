
################################
####   Training Inducible   ####
################################

from gpro.generator.others.cgan.cgan import Deepseed

model = Deepseed(n_iters=10000, save_iters=10000, dataset="./datasets/ecoli_mpra_3_laco.csv", savepath="./checkpoints")
model.train()

model.generate(input_file = './datasets/input_promoters.txt', sample_model_path='./checkpoints/check/deepseed_ecoli_mpra_3_laco/net_G_9999.pth')



###################################
####   Training Constitutive   ####
###################################

from gpro.generator.others.cgan.cgan import Deepseed

model = Deepseed(n_iters=3000, save_iters=500, dataset="./datasets/ecoli_mpra_-10_-35.csv", savepath="./checkpoints",
                 data_name = "ecoli_mpra_-10_-35", model_name = "deepseed_ecoli_mpra_-10_-35")
model.train()


#############################
####   Displaying Whole  ####
#############################


import sys
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import collections
from utils.seqdeal import kmer_count, remove_1035, get_kmer_stat


MEDIUM_SIZE = 8
SMALLER_SIZE = 6
BIGGER_SIZE = 25
plt.rc('font', family='Helvetica', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	# fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 2
FIG_WIDTH = 2

k_size = 4   # set k-mer k number
mpra_consituitive_path = "./datasets/ecoli_mpra_-10_-35.csv"  # control group dataset
gen_data_path = "./checkpoints/cache/deepseed_ecoli_mpra_-10_-35/inducible/inducible_ecoli_mpra_-10_-35_2023-09-30-02-04-04_results.csv"  # generation group dataset

# Get the original data
mpra_data, gen_data = pd.read_csv(mpra_consituitive_path), pd.read_csv(gen_data_path)

# Remove -10 and -35 sequences
mpra_flanking, gen_flanking = remove_1035(mpra_data['realB'], mpra_data['realA']), \
                                remove_1035(gen_data['fakeB'], gen_data['realA'])

# Get the kmer statistics
kmer_stat_mpra, kmer_stat_gen = get_kmer_stat(gen_flanking, mpra_flanking, k_size)

# Calculate the pearson correlation
coefs = pearsonr(list(kmer_stat_mpra.values()), list(kmer_stat_gen.values()))
print('Coefficients: {}'.format(coefs))

# Plot the scatter plot
fig, ax = plt.subplots(figsize=(1.2 * FIG_WIDTH * 1.2, FIG_HEIGHT))
plt.scatter(list(kmer_stat_mpra.values()), list(kmer_stat_gen.values()), s=0.3, c='#2d004b')
csfont = {'family': 'Helvetica'}
plt.setp(ax.get_xticklabels(), rotation=0, ha="center", va="top",
          rotation_mode="anchor")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
ax.set_aspect(1 * abs(x1 - x0) / abs(y1 - y0))
ax.set_title('{}-mer frequency of PccGEO and Natural promoters'.format(k_size), fontdict=csfont)
ax.set_xlabel('Natural', fontdict=csfont)
ax.set_ylabel('PccGEO', fontdict=csfont)
fig.tight_layout()

# Save the figure
plt.savefig('./results/PccGEO_{}mer_scatter.pdf'.format(k_size))


