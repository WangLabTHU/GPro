

######################
####   Predictor   ###
######################

from gpro.predictor.others.GRUClassifier import GRUClassifier_language
from gpro.predictor.cnn_k15.cnn_k15 import CNN_K15_language
from gpro.evaluator.regression import plot_regression_performance

model = GRUClassifier_language(length=306)

dataset = './datasets/predictor_seq.txt'
labels = './datasets/predictor_cls.txt'
save_path = './checkpoints/'
model.train(dataset=dataset,labels=labels,savepath=save_path)


######################
####   Feedback   ####
######################

from gpro.generator.diffusion.diffusion import Diffusion_language
from gpro.optimizer.model_driven.feedback import Feedback

generator = Diffusion_language(length=156, transformer_local_size=39, dataset_type="padding", epochs=20, check_every=20)
predictor = GRUClassifier_language(length=306)
predictor_modelpath = "./checkpoints/GRUClassifier/checkpoint.pth"
natural_datapath = "./datasets/generator_seq.txt"

tmp = Feedback(generator=generator, predictor=predictor, 
                   predictor_modelpath=predictor_modelpath, sample_number=1000,
                   natural_datapath=natural_datapath, savepath="./results/Feedback")
tmp.run(mode="diffusion", MaxEpoch=5, MaxIter=50, MaxPoolsize=200)


######################
#####   Direct   #####
######################

from gpro.generator.diffusion.diffusion import Diffusion_language

dataset_path = "./generator_seq.txt"
checkpoint_path = "./checkpoints"
generator = Diffusion_language(length=156, transformer_local_size=39, dataset_type="padding", epochs=100, check_every=20)
generator.train(dataset=dataset_path, savepath=checkpoint_path)

generator_modelpath = "./checkpoints/diffusion/check/checkpoint.pt"
generator.generate(generator_modelpath, 3655)


###########################
#####   Displaying    #####
###########################

import Levenshtein
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from gpro.utils.base import open_fa, write_profile, write_fa, write_seq

def transitional_levenshtein_distance_calculation(samples_path1, samples_path2):
    samples1 = open_fa(samples_path1)
    samples2 = open_fa(samples_path2)
    
    samples_length = len(samples1)
    
    res_list = []
    for i in range(samples_length):
        seq_sim = []
        for j in range(samples_length):
            dis = Levenshtein.distance(samples1[i],samples2[j])
            sim = 1 - dis / len(samples1[i])
            seq_sim.append(sim)
        res_list.append(max(seq_sim))
    return res_list



file_natural = "./datasets/generator_seq.txt"
file_direct = "./checkpoints/diffusion/samples/sample_ep100_s0_num_3655.txt"
file_feedback = "./results/Feedback/traj/ExpIter_49.txt"
sim_direct = transitional_levenshtein_distance_calculation(file_direct, file_natural)
sim_feedback = transitional_levenshtein_distance_calculation(file_feedback, file_natural)


font = {'size' : 10}
matplotlib.rc('font', **font)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
        
vmin = min(min(sim_direct), min(sim_feedback))
vmax = max(max(sim_direct), max(sim_feedback))
sns.histplot(sim_direct, label='Before feedback', alpha = 0.4, bins=50, kde=True,  binrange=(vmin,vmax), stat="density", linewidth=0)
sns.histplot(sim_feedback, label='After feedback', alpha = 0.4, bins=50, kde=True,  binrange=(vmin,vmax), stat="density", linewidth=0)

ax.set_xlabel('Normalized edit distance', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(loc='upper right')
plt.title("")
plt.show()
plt.savefig("./results/normalize_distance.png")

