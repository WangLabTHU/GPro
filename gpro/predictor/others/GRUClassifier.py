import os
import sys
import functools
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from gpro.utils.utils_predictor import EarlyStopping, open_fa, open_exp


'''
GRU is not limited by length but is subject to batch_size limit
'''

def seq2onehot(seq):
        module = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        i = 0
        promoter_onehot = []
        while i < len(seq):
           tmp = []
           for item in seq[i]:
                if item == 't' or item == 'T':
                    tmp.append(module[0])
                elif item == 'c' or item == 'C':
                    tmp.append(module[1])
                elif item == 'g' or item == 'G':
                    tmp.append(module[2])
                elif item == 'a' or item == 'A':
                    tmp.append(module[3])
                else:
                    tmp.append([0,0,0,0])
           promoter_onehot.append(np.array(tmp))
           i = i + 1      
        return np.array(promoter_onehot)


def padding(data):
    length = []
    for i in range(len(data)):
        length.append(data[i].shape[0])
    max_len = max(length)
    
    new_data = []
    for item in data:
        pad = torch.tensor([[0,0,0,0]] * (max_len - item.shape[0]))
        new_item = torch.cat((item, pad), 0)
        new_data.append(new_item)
    new_data = torch.tensor([item.cpu().detach().numpy() for item in new_data])
    device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"), ]
    new_data = new_data.to(device)
    return new_data
            

class GRUClassifier(nn.Module):

    def __init__(self, vocab_size=4, batch_size=64, hidden_dim=128):
        super(GRUClassifier, self).__init__()
        self.hidden = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = 1
        
        self.embedding = nn.Linear(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=False, num_layers=self.num_layers, dropout=0.3)
        self.linear = nn.Linear(hidden_dim, 1)
        self.batch_size = batch_size
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.gru(x)
        x = F.sigmoid(self.linear(x[-1]))
        return x, h

    def init_hidden(self):
        hidden = torch.randn(self.num_layers, self.batch_size, self.hidden)
        if self.use_cuda:
            device = torch.device("cuda")
            return hidden.to(device)
        return hidden


class GRUClassifier_language:
    def __init__(self, 
                 length,
                 batch_size = 64,
                 model_name = "GRUClassifier",
                 epoch = 200,
                 patience = 10,
                 log_steps = 10,
                 save_steps = 20,
                 exp_mode = "direct"
                 ):
      
        self.model = GRUClassifier(batch_size=batch_size)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience
        self.seq_len = length
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"), ]
        self.exp_mode = exp_mode
    
    def train(self, dataset, labels, savepath):
      
        self.dataset = dataset
        self.labels = labels
        self.checkpoint_root = savepath
      
        filename_sim = self.checkpoint_root + self.model_name
        
        if not os.path.exists(filename_sim):
            os.makedirs(filename_sim)
        
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, 
                                       path=os.path.join(filename_sim, 'checkpoint.pth'), stop_order='max')
        
        total_feature = open_fa(self.dataset)
        total_feature = seq2onehot(total_feature)
        total_label = open_exp(self.labels, operator=self.exp_mode)
        total_feature = [torch.tensor(item, dtype=float) for item in total_feature] # (sample num,length,4)
        total_label = [torch.tensor(item, dtype=float) for item in total_label] # (sample num)
        
        total_length = len(total_feature)
        r = int(total_length*0.7)
        train_feature = total_feature[0:r]
        train_label = total_label[0:r]
        valid_feature = total_feature[r:total_length]
        valid_label = total_label[r:total_length]
        
        train_log_filename = os.path.join(filename_sim, "train_log.txt")
        train_model_filename = os.path.join(filename_sim, "checkpoint.pth")
        print("results saved in: ", filename_sim)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),] 
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.BCELoss()
        
        num_batches_train = int(len(train_feature)/self.batch_size)
        num_batches_valid = int(len(valid_feature)/self.batch_size)
        
        h = self.model.init_hidden()
        v = self.model.init_hidden()
        for epoch in tqdm(range(0,self.epoch)):
            model.train()
            train_epoch_loss = []
            for batch in range(num_batches_train): # torch.Size([batch_size, seq_len, 4])
                feature = train_feature[batch*self.batch_size:(batch+1)*self.batch_size]
                label = train_label[batch*self.batch_size:(batch+1)*self.batch_size]
                feature = padding(feature)
                label = torch.tensor(label)
                
                feature = feature.to(torch.float32).to(device).permute(1,0,2) # [batch_size, 4, seq_len]
                label = label.to(torch.float32).to(device)
                h.detach_()
                outputs, h = model(feature, h)
                optimizer.zero_grad()
                
                loss = criterion(outputs.flatten(), label.float())
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())

            model.eval()
            valid_exp_real = []
            valid_exp_pred = []
            for batch in range(num_batches_valid): # torch.Size([batch_size, seq_len, 4])
                feature = valid_feature[batch*self.batch_size:(batch+1)*self.batch_size]
                label = valid_label[batch*self.batch_size:(batch+1)*self.batch_size]
                feature = padding(feature)
                label = torch.tensor(label)
                
                feature = feature.to(torch.float32).to(device).permute(1,0,2)
                label = label.to(torch.float32).to(device)
                outputs, v = model(feature, v)
                valid_exp_real += label.float().tolist()
                valid_exp_pred += outputs.flatten().tolist()
            coefs = np.corrcoef(valid_exp_real,valid_exp_pred)
            coefs = coefs[0, 1]
            test_coefs = coefs
            
            print("real expression samples: ", valid_exp_real[0:5])
            print("pred expression samples: ", valid_exp_pred[0:5])
            print("current coeffs: ", test_coefs)
            cor_pearsonr = pearsonr(valid_exp_real, valid_exp_pred)
            print("current pearsons: ",cor_pearsonr)
            
            ## Early Stopping Step
            early_stopping(val_loss=test_coefs, model=self.model)
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
        
        test_feature = seq2onehot(test_feature)
        test_feature = [torch.tensor(item, dtype=float) for item in test_feature]
        
        num_batches_test = int(len(test_feature)/self.batch_size)
        h = self.model.init_hidden()
        
        test_exp_pred = []
        for batch in range(num_batches_test): # torch.Size([batch_size, seq_len, 4])
            feature = test_feature[batch*self.batch_size:(batch+1)*self.batch_size]
            feature= padding(feature)

            feature = feature.to(torch.float32).to(device).permute(1,0,2)
            outputs, h = model(feature, h)
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
            test_feature = seq2onehot(test_feature)
        elif mode=="data":
            test_feature = seq2onehot(inputs)
        elif mode=="onehot":
            test_feature = inputs
        test_feature = [torch.tensor(item, dtype=float) for item in test_feature]
        
        num_batches_test = int(len(test_feature)/self.batch_size)
        h = self.model.init_hidden()
        
        test_exp_pred = []
        for batch in range(num_batches_test): # torch.Size([batch_size, seq_len, 4])
            feature = test_feature[batch*self.batch_size:(batch+1)*self.batch_size]
            feature = padding(feature)

            feature = feature.to(torch.float32).to(device).permute(1,0,2)
            outputs, h = model(feature, h)
            pred = outputs.flatten().tolist()
            test_exp_pred += pred
            
        return test_exp_pred

