import argparse
import numpy as np
import pandas as pd
from gpro.generator.wgan.wgan import WGAN_language
from gpro.predictor.cnn_k15.cnn_k15 import CNN_K15_language
from gpro.optimizer.model_driven.filter import Filter
from gpro.optimizer.model_driven.gradient import GradientAlgorithm

from gpro.evaluator.kmer import plot_kmer
from gpro.evaluator.mutagenesis import plot_mutagenesis
from gpro.evaluator.regression import plot_regression_performance
from gpro.evaluator.saliency import plot_saliency_map
from gpro.evaluator.seqlogo import plot_seqlogos

from gpro.utils.base import *

################################
####   Generator Training   ####
################################

print("\n Step1: Start Generator Training\n")

parser = argparse.ArgumentParser(description="A simple use case of Gpro package.")
parser.add_argument('-seq', type=str, help='seqs for model training, all input sequences should have the same length!', default='./extdata/seq.txt')
parser.add_argument('-exp', type=str, help='expression for model training, corresponding with seq file.', default = './extdata/exp.txt')
parser.add_argument('-length', type=int, help='sequence length', default=50)
args = parser.parse_args()

seq_path = args.seq
exp_path = args.exp
seq_len = args.length
checkpoint_path = './checkpoints/'

generator = WGAN_language(length=seq_len, num_epochs=12, print_epoch = 12, save_epoch = 12)
generator.train(dataset=seq_path, savepath=checkpoint_path)

print("\n Generator training finish! Model has been saved in ./checkpoints/wgan \n")


################################
####   Predictor Training   ####
################################

print("\n Step2: Start Predictor Training\n")

predictor = CNN_K15_language(length=seq_len, epoch=200, patience=50, exp_mode="direct")
predictor.train(dataset=seq_path,labels=exp_path,savepath=checkpoint_path)

print("\n Predictor training finish! Model has been saved in ./checkpoints/cnn_k15 \n")

################################
####        Selecting       ####
################################

generator_modelpath = './checkpoints/wgan/checkpoints/net_G_12.pth'
predictor_modelpath = './checkpoints/cnn_k15/checkpoint.pth'
natural_datapath = seq_path

print("\n Step3.1: Start Directly Selecting New Sequences \n")

tmp = Filter(generator=generator, predictor = predictor, 
             generator_modelpath=generator_modelpath, predictor_modelpath=predictor_modelpath,
             natural_datapath=natural_datapath, savepath="./optimization/Filter")

tmp.run(MaxEpoch=100)


print("\n Step3.2: Start Performing Gradient-based Optimization \n")



tmp = GradientAlgorithm(predictor = predictor, 
                        generator_modelpath=generator_modelpath, predictor_modelpath=predictor_modelpath,
                        natural_datapath=natural_datapath, savepath="./optimization/Gradient")

tmp.run()

print("\n Optimization finish! Result has been saved in ./optimization/ \n")
sample_path = "./checkpoints/wgan/samples/sample_ep12_s0_num_1000.txt"

################################
####        Analysing       ####
################################

seqs = open_fa(seq_path)
pred = predictor.predict_input(model_path=predictor_modelpath, inputs=seq_path)

if not os.path.exists("./evaluation"):
    os.makedirs("./evaluation")
write_seq("./evaluation/seqs.txt", seqs)
write_exp("./evaluation/pred.txt", pred)


################################
####       Evaluating       ####
################################

print("\n Step4: Start Evaluating \n")

generator_training_datapath = seq_path
generator_sampling_datapath = sample_path
predictor_training_datapath = seq_path
predictor_expression_datapath = exp_path
predictor_prediction_datapath = "./evaluation/pred.txt"

plot_kmer(generator_training_datapath, generator_sampling_datapath, report_path="./evaluation/", file_tag="WGAN")
plot_mutagenesis(predictor, predictor_modelpath, predictor_training_datapath, predictor_expression_datapath,
                     report_path="./evaluation/", file_tag="CNNK15")
metrics = plot_regression_performance( predictor_expression_datapath, predictor_prediction_datapath,
                                          report_path="./evaluation/", file_tag="CNNK15")
print("ad_mean_Y: {}, ad_std:{}, ad_r2:{}, ad_pearson:{}, ad_spearman:{} \n".format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))

plot_saliency_map(predictor, predictor_training_datapath, predictor_modelpath, 
                      report_path="./evaluation/", file_tag="CNNK15")
plot_seqlogos(predictor, predictor_training_datapath, predictor_modelpath, 
                    report_path="./evaluation/", file_tag="CNNK15")



print("\n Evaluating finish! Result has been saved in ./evaluation/ \n")