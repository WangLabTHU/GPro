
from gpro.generator.wgan.wgan import WGAN_language
from gpro.evaluator.kmer import plot_kmer_with_model, plot_kmer

######################
####   Training   ####
######################

## Step 1

dataset_path = './datasets/sequence_data.txt'
checkpoint_path = './checkpoints'
model = WGAN_language(length=50, num_epochs=12, print_epoch = 12, save_epoch = 12)
model.train(dataset=dataset_path, savepath=checkpoint_path)

## Step 2

model = WGAN_language(length=50)
sample_modelpath = "./checkpoints/wgan/checkpoints/net_G_12.pth"
sample_number = 10000
model.generate(sample_modelpath, sample_number)

###########################
#####   Displaying    #####
###########################

## Step 3

generator = WGAN_language(length=50)
generator_training_datapath = './datasets/sequence_data.txt'
generator_modelpath = "./checkpoints/wgan/checkpoints/net_G_12.pth"
plot_kmer_with_model(generator, generator_modelpath,  generator_training_datapath,
                         report_path="./results/", file_tag="WGAN_01")

## Step 4

generator_training_datapath = './datasets/sequence_data.txt'
generator_sampling_datapath = './checkpoints/wgan/samples/sample_ep12_s0_num_10000.txt' 
plot_kmer(generator_training_datapath, generator_sampling_datapath, report_path="./results/", file_tag="WGAN_02")

