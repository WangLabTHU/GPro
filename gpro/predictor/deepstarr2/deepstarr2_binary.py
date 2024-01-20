import os
import sys
import functools
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

import torch
from torch import nn
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from ...utils.utils_predictor import EarlyStopping, seq2onehot, open_fa, open_exp

from sklearn.metrics import accuracy_score, precision_recall_curve, auc

class SequenceData(Dataset):
  def __init__(self,data, label):
    self.data = data
    self.target = label
  
  def __getitem__(self, index):
    return self.data[index], self.target[index]
    
  def __len__(self):
    return self.data.size(0)
  
  def __getdata__(self):
    return self.data, self.target

class TestData(Dataset):
    def __init__(self,data):
        self.data = data
  
    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return self.data.size(0)
    
    def __getdata__(self):
        return self.data

def open_binary(path):
    record = open_fa(path)
    new_record = [1 if status == 'Active' else 0 for status in record]
    return new_record

def proba_to_label(data):
    new_data = [1 if num > 0.5 else 0 for num in data]
    return new_data

class DeepSTARR2_binary(nn.Module):
    def __init__(self, kernel_size = [7, 3, 3, 3], filter_num = [256, 120, 60, 60], poolsize = 3,
                 n_conv_layer = 4, n_add_layer = 2, neuron_dense = [64, 256], dropout_rate=0.4, input_length = 1001):
        super(DeepSTARR2_binary, self).__init__()

        conv = []
        filter_in = 4
        output_length = input_length
        for i in range(n_conv_layer):
            conv.append( nn.Conv1d(in_channels=filter_in, out_channels=filter_num[i], kernel_size=kernel_size[i], padding='same') )
            conv.append( nn.BatchNorm1d(filter_num[i]) )
            conv.append( nn.ReLU() )
            conv.append( nn.MaxPool1d(kernel_size=poolsize) )
            filter_in = filter_num[i]
            output_length = int(output_length / poolsize)
        self.conv = nn.Sequential(*conv)
        self.flatten = nn.Flatten()
        
        dense = []
        linear_in = output_length * filter_num[-1]
        for i in range(n_add_layer):
            dense.append( nn.Linear(linear_in, neuron_dense[i]) )
            dense.append( nn.BatchNorm1d(neuron_dense[i]) )
            dense.append( nn.ReLU() )
            dense.append( nn.Dropout(dropout_rate) )
            linear_in = neuron_dense[i]
        self.dense = nn.Sequential(*dense)
        
        ## binary
        self.newlinear = nn.Linear(neuron_dense[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.flatten(x) 
        x = self.dense(x) 
        x = self.newlinear(x) 
        output = self.sigmoid(x) 
        return output

class DeepSTARR2_binary_language:
    def __init__(self, 
                 length,
                 batch_size = 64,
                 model_name = "deepstarr2_binary",
                 epoch = 200,
                 patience = 50,
                 log_steps = 10,
                 save_steps = 20,
                 ):   
        self.model = DeepSTARR2_binary(input_length=length)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience
        self.seq_len = length
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"), ]
    
    def train_with_valid(self, train_dataset, train_labels, valid_dataset, valid_labels, savepath, transfer=False, modelpath=None):
        
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.checkpoint_root = savepath
        self.transfer = transfer
        
        if(self.transfer):
            pretrained_model = torch.load(modelpath)
            model_dict = self.model.state_dict()
            state_dict = {k:v for k,v in pretrained_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        
        filename_sim = self.checkpoint_root + self.model_name
        
        if not os.path.exists(filename_sim):
            os.makedirs(filename_sim)
        
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, 
                                       path=os.path.join(filename_sim, 'checkpoint.pth'), stop_order='max')
        
        train_feature = open_fa(self.train_dataset)
        train_feature = seq2onehot(train_feature, self.seq_len)
        train_label = open_binary(self.train_labels)
        train_feature = torch.tensor(train_feature, dtype=float) # (sample num,length,4)
        train_label = torch.tensor(train_label, dtype=float) # (sample num)

        valid_feature = open_fa(self.valid_dataset)
        valid_feature = seq2onehot(valid_feature, self.seq_len)
        valid_label = open_binary(self.valid_labels)
        valid_feature = torch.tensor(valid_feature, dtype=float) # (sample num,length,4)
        valid_label = torch.tensor(valid_label, dtype=float) # (sample num)
        
        train_dataset = SequenceData(train_feature, train_label)
        train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size, shuffle=True)
        valid_dataset = SequenceData(valid_feature, valid_label)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=self.batch_size, shuffle=True)
        
        train_log_filename = os.path.join(filename_sim, "train_log.txt")
        train_model_filename = os.path.join(filename_sim, "checkpoint.pth")
        print("results saved in: ", filename_sim)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),] 
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.BCELoss(reduction="mean")
        
        for epoch in tqdm(range(0,self.epoch)):
            model.train()
            train_epoch_loss = []
            for idx,(feature,label) in enumerate(train_dataloader,0):
                feature = feature.to(torch.float32).to(device).permute(0,2,1)
                label = label.to(torch.float32).to(device)
                outputs = model(feature)
                optimizer.zero_grad()
                
                loss = criterion(outputs.flatten(), label.float())
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())

            model.eval()
            valid_exp_real = []
            valid_exp_pred = []
            for idx,(feature,label) in enumerate(valid_dataloader,0):
                feature = feature.to(torch.float32).to(device).permute(0,2,1)
                label = label.to(torch.float32).to(device)
                outputs = model(feature)
                valid_exp_real += label.float().tolist()
                valid_exp_pred += outputs.flatten().tolist()
            
            print("real expression samples: ", valid_exp_real[0:5])
            print("pred expression samples: ", valid_exp_pred[0:5])
            
            precision, recall, _ = precision_recall_curve(valid_exp_real, valid_exp_pred)
            prauc = auc(recall, precision)
            valid_exp_pred = proba_to_label(valid_exp_pred)
            acc = accuracy_score(valid_exp_real, valid_exp_pred)
            
            print("current PR-AUC: ", prauc)
            print("current accuracy: ", acc)
            
            ## Early Stopping Step
            early_stopping(val_loss=acc, model=self.model)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break
            
            if (epoch%self.log_steps == 0):
                to_write = "epoch={}, loss={}\n".format(epoch, np.average(train_epoch_loss))
                with open(train_log_filename, "a") as f:
                    f.write(to_write)
            if (epoch%self.save_steps == 0):
                torch.save(model.state_dict(), train_model_filename)
    
    def predict(self, model_path, data_path):
        
        model_path = os.path.dirname(model_path)
        path_check = '{}/checkpoint.pth'.format(model_path)
        path_seq_save =  '{}/seqs.txt'.format(model_path)
        path_pred_save = '{}/preds.txt'.format(model_path)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
        model = self.model.to(device)
        model.load_state_dict(torch.load(path_check))
        model.eval()
        seq_len = self.seq_len
        
        test_feature = open_fa(data_path)
        test_seqs = test_feature
        
        test_feature = seq2onehot(test_feature, seq_len)
        test_feature = torch.tensor(test_feature, dtype=float)
        test_dataset = TestData(test_feature)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = 128, shuffle=False)
        
        test_exp_pred = []
        for idx,feature in enumerate(test_dataloader,0):
            feature = feature.to(torch.float32).to(device).permute(0,2,1)
            outputs = model(feature)
            pred = outputs.flatten().tolist()
            test_exp_pred += pred
        
        ## Saving Seqs
        f = open(path_seq_save,'w')
        i = 0
        while i < len(test_seqs):
            f.write('>' + str(i) + '\n')
            f.write(test_seqs[i] + '\n')
            i = i + 1
        f.close()
        
        ## Saving pred exps
        f = open(path_pred_save,'w')
        i = 0
        while i < len(test_exp_pred):
            f.write(str(np.round(test_exp_pred[i],2)) + '\n')
            i = i + 1
        f.close()

    def predict_input(self, model_path, inputs, mode="path"):
        
        model_path = os.path.dirname(model_path)
        path_check = '{}/checkpoint.pth'.format(model_path)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
        model = self.model.to(device)
        model.load_state_dict(torch.load(path_check))
        model.eval()
        seq_len = self.seq_len
        
        if mode=="path":
            test_feature = open_fa(inputs)
            test_feature = seq2onehot(test_feature, seq_len)
        elif mode=="data":
            test_feature = seq2onehot(inputs, seq_len)
        elif mode=="onehot":
            test_feature = inputs
        test_feature = torch.tensor(test_feature, dtype=float)
        test_dataset = TestData(test_feature)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = 128, shuffle=False)
        
        exp = []
        for idx,feature in enumerate(test_dataloader,0):
            feature = feature.to(torch.float32).to(device).permute(0,2,1)
            outputs = model(feature)
            pred = outputs.flatten().tolist()
            exp += pred
        return exp
    
    # access
    def predict_without_model(self, modelpath, inputs, mode="path"):
        pretrained_model = torch.load(modelpath)
        model_dict = self.model.state_dict()
        state_dict = {k:v for k,v in pretrained_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
        model = self.model.to(device)
        model.eval()
        seq_len = self.seq_len
        
        if mode=="path":
            test_feature = open_fa(inputs)
            test_feature = seq2onehot(test_feature, seq_len)
        elif mode=="data":
            test_feature = seq2onehot(inputs, seq_len)
        elif mode=="onehot":
            test_feature = inputs
        test_feature = torch.tensor(test_feature, dtype=float)
        test_dataset = TestData(test_feature)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = 128, shuffle=False)
        
        exp = []
        for idx,feature in enumerate(test_dataloader,0):
            feature = feature.to(torch.float32).to(device).permute(0,2,1)
            outputs = model(feature)
            pred = outputs.flatten().tolist()
            exp += pred
        return exp