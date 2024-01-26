from gpro.predictor.deepstarr2.deepstarr2 import DeepSTARR2_language
from gpro.predictor.deepstarr2.deepstarr2_binary import DeepSTARR2_binary_language, open_binary
from scipy.stats import pearsonr
from gpro.utils.utils_predictor import EarlyStopping, seq2onehot, open_fa, open_exp
from gpro.evaluator.regression import plot_regression_performance
from sklearn.metrics import accuracy_score, precision_recall_curve, auc

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

##########################################
####   Training Accessibility Model   ####
##########################################

# model = DeepSTARR2_language(length=1001, epoch=200, patience=20, model_name="deepstarr2_fold9")
# train_dataset = "./datasets/Accessibility_models_training_data/Train_seq.txt"
# train_labels  = "./datasets/Accessibility_models_training_data/Train_exp_epidermis.txt"
# valid_dataset = "./datasets/Accessibility_models_training_data/Val_seq.txt"
# valid_labels = "./datasets/Accessibility_models_training_data/Val_exp_epidermis.txt"
# save_path = "./checkpoints/"
# model.train_with_valid(train_dataset=train_dataset,train_labels=train_labels,valid_dataset=valid_dataset, valid_labels=valid_labels, savepath=save_path)

# model = DeepSTARR2_language(length=1001, epoch=200, patience=20)
# model_path = "./checkpoints/deepstarr2_fold9/checkpoint.pth"
# data_path =  "./datasets/Accessibility_models_training_data/Test_seq.txt"
# model.predict(model_path=model_path, data_path=data_path)

##########################################
####   Testing Accessibility Model   ####
##########################################

predictor_prediction_datapath = "./checkpoints/deepstarr2_fold9/preds.txt"
predictor_expression_datapath = "./datasets/Accessibility_models_training_data/Test_exp_epidermis.txt"

metrics = plot_regression_performance( predictor_expression_datapath, predictor_prediction_datapath,
                                          report_path="./results/", file_tag="DeepSTARR2")
print("ad_mean_Y: {}, ad_std:{}, ad_r2:{}, ad_pearson:{}, ad_spearman:{} \n".format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))

## ad_mean_Y: 0.6905912721732207, ad_std:0.5287924997326767, ad_r2:0.4894610339334141, ad_pearson:0.6996149183182232, ad_spearman:0.6898636577449109

########################################################
####   Training Activity Model: Transfer Learning   ####
########################################################

# model = DeepSTARR2_binary_language(length=1001, epoch=200, patience=20, model_name="deepstarr2_binary")
# train_dataset = "./datasets/EnhancerActivity_models_training_data/Train_seq.txt"
# train_labels  = "./datasets/EnhancerActivity_models_training_data/Train_exp_epidermis.txt"
# valid_dataset = "./datasets/EnhancerActivity_models_training_data/Val_seq.txt"
# valid_labels = "./datasets/EnhancerActivity_models_training_data/Val_exp_epidermis.txt"
# save_path = "./checkpoints/"
# model_path = "./checkpoints/deepstarr2_fold9/checkpoint.pth"
# model.train_with_valid(train_dataset=train_dataset,train_labels=train_labels,valid_dataset=valid_dataset, valid_labels=valid_labels, savepath=save_path,
#                        transfer=True, modelpath=model_path)


#####################################################
####   Training Activity Model: Random Initial   ####
#####################################################

# model = DeepSTARR2_binary_language(length=1001, epoch=200, patience=20, model_name="deepstarr2_random")
# train_dataset = "./datasets/EnhancerActivity_models_training_data/Train_seq.txt"
# train_labels  = "./datasets/EnhancerActivity_models_training_data/Train_exp_epidermis.txt"
# valid_dataset = "./datasets/EnhancerActivity_models_training_data/Val_seq.txt"
# valid_labels = "./datasets/EnhancerActivity_models_training_data/Val_exp_epidermis.txt"
# save_path = "./checkpoints/"
# model.train_with_valid(train_dataset=train_dataset,train_labels=train_labels,valid_dataset=valid_dataset, valid_labels=valid_labels, savepath=save_path)

#######################################################
####   Testing Activity Model: Transfer Learning   ####
#######################################################

model = DeepSTARR2_binary_language(length=1001, epoch=200, patience=20)
model_path = "./checkpoints/deepstarr2_binary/checkpoint.pth"
data_path =  "./datasets/EnhancerActivity_models_training_data/Test_seq.txt"
pred_transfer = model.predict_input(model_path=model_path, inputs=data_path)
expr_transfer = open_binary("./datasets/EnhancerActivity_models_training_data/Test_exp_epidermis.txt")

precision_transfer, recall_transfer, _ = precision_recall_curve(expr_transfer, pred_transfer)
prauc_transfer = auc(recall_transfer, precision_transfer)
print("predicted prauc for transfer learning: ",prauc_transfer)


#####################################################
####   Testing Activity Model: Random Initial   #####
#####################################################

model = DeepSTARR2_binary_language(length=1001, epoch=200, patience=20)
model_path = "./checkpoints/deepstarr2_random/checkpoint.pth"
data_path =  "./datasets/EnhancerActivity_models_training_data/Test_seq.txt"
pred_random = model.predict_input(model_path=model_path, inputs=data_path)
expr_random = open_binary("./datasets/EnhancerActivity_models_training_data/Test_exp_epidermis.txt")

precision_random, recall_random, _ = precision_recall_curve(expr_random, pred_random)
prauc_random = auc(recall_random, precision_random)
print("predicted prauc for random initialization: ",prauc_random)

#####################################################
####   Testing Activity Model: Driect Access   ######
#####################################################

model = DeepSTARR2_binary_language(length=1001, epoch=200, patience=20)
model_path = "./checkpoints/deepstarr2_fold9/checkpoint.pth"
data_path =  "./datasets/EnhancerActivity_models_training_data/Test_seq.txt"
pred_access = model.predict_without_model(modelpath=model_path, inputs=data_path)
expr_access = open_binary("./datasets/EnhancerActivity_models_training_data/Test_exp_epidermis.txt")

precision_access, recall_access, _ = precision_recall_curve(expr_access, pred_access)
prauc_access = auc(recall_access, precision_access)
print("predicted prauc without second round training: ",prauc_access)

###############################################
####   Figure Plot: Precision-Recall Curve ####
###############################################

sns.set_style("darkgrid")

font = {'size' : 12}
matplotlib.rc('font', **font)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
fig, ax = plt.subplots(figsize = (8,8), dpi = 600)

sns.lineplot(x=recall_transfer, y=precision_transfer, label="Pred. Enh. act. (AUC: {})".format(round(prauc_transfer, 4)), color="Crimson")
sns.lineplot(x=recall_random, y=precision_random, label="Pred. Enh. act.(random init.) (AUC: {})".format(round(prauc_random, 4)), color="SlateGrey")
sns.lineplot(x=recall_access, y=precision_access, label="Pred. Access. (AUC: {})".format(round(prauc_access, 4)), color="RoyalBlue")

ax.set_xlabel('Recall', fontsize=14)
ax.set_ylabel('Precision', fontsize=14)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xticks(np.arange(0, 1, 0.2))
ax.set_yticks(np.arange(0, 1, 0.2))

plt.title("")
plt.savefig("./results/PR_epidermis_fold9.png")

# predicted prauc for transfer learning:  0.5313295253334795
# predicted prauc for random initialization:  0.5172747227557057
# predicted prauc without second round training:  0.35590014047412005


