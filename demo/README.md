
Demo1: Eukaryotic

***The evolution, evolvability and engineering of gene regulatory DNA*** 

original model: https://github.com/1edv/evolution/blob/master/manuscript_code/model/tpu_model/

datasets are generated from training_data_Glu.txt (https://codeocean.com/capsule/8020974/tree/v1), from 29G data we selected 100k sequences for training.

We have reproduced Figure 1b(Pearson correlation coefficient graph) of the original paper. 

We have also fulfilled the SSWM algorithm through our optimizer, reproduced the Figure 2f(Boxplot of two SSWM mode).

Demo2: Prokaryote

***Synthetic promoter design in Escherichia coli based on a deep generative network*** 

original model: https://github.com/HaochenW/Deep_promoter/tree/master

datasets are obtained from sequence_data.txt, using all 14000+ sequences for wgan generation.

We have reproduced Figure S3(k-mer estimation) of the original paper.

Demo3: 1000bp regulatory sequences

***Controlling gene expression with deep generative design of regulatory DNA***

orginal figure: https://www.nature.com/articles/s41467-022-32818-8/figures/3

datasets are obtained from https://zenodo.org/record/6811226, from scerevisiae.rsd1.lmbda_22.1000.npz dataset.

We have reproduced the gradient optimization with our model, and reproduced Figure 3b of the original paper.

We did not obtain the data for tpm, so we take log2 rather than log10, considering the difference of 10-15 times between the average predicted value and tpm.

Demo4: in silico simulation

***Feedback GAN for DNA optimizes protein functions***

original figure: https://www.nature.com/articles/s42256-019-0017-4/figures/4

datasets are obtained from https://github.com/av1659/fbgan/tree/master/data

We have reproduced the Figure 4a(normalized edited distance), with Diffusion from Generator and Feedback from Optimizer.

There are some differences in the results, but the trend is consistent.

Demo5: conditional GAN

***Deep flanking sequence engineering for efficient promoter design***

We have reproduced figure 2c from: https://www.biorxiv.org/content/10.1101/2023.04.14.536502v1


***Targeted design of synthetic enhancers for selected tissues in the Drosophila embryo***

We have reproduced the Extended Data Fig. 1b and Extended Data Fig. 3b

datasets are obtained from https://zenodo.org/records/8011697

Due to the large size of the dataset, we have only provided the most necessary data. For detailed information, please download it from Zonedo and refer to the data processing method under '/data'.

We have provided all needed checkpoints for this demo, and annotated out all the training steps. You can directly reproduce the codes from our pretrained models.