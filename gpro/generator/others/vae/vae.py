import os,sys
import numpy as np
from tqdm import tqdm
from time import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from .common import Seq_loader, onehot2seq, sequence2fa, seq2onehot

'''
input should be [batch_size, seq_len, n_tokens]
'''

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Resblock(nn.Module):
    def __init__(self, kernel_size=20, model_dim = 256):
        super(Resblock, self).__init__()
        self.model_dim = model_dim
        self.normal_way = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(self.model_dim, self.model_dim, kernel_size=kernel_size, padding="same", bias=True),
            nn.ReLU(inplace=False),
            nn.Conv1d(self.model_dim, self.model_dim, kernel_size=kernel_size, padding="same", bias=True))
    def forward(self, inputs):
        return 0.3 * self.normal_way(inputs) + inputs


class VAE(nn.Module):
    def __init__(self, n_tokens = 4, hidden_dim = 256, latent_dim = 64, seq_len = 80):
        super(VAE, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_layers = 5
        self.motif_conv_hidden = hidden_dim
        self.conv_width_motif = 30
        
        # Build Encoder
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.motif_conv_hidden, kernel_size=self.conv_width_motif, padding='same')
        self.resblock11 = Resblock()
        self.resblock12 = Resblock()
        self.resblock13 = Resblock()
        self.resblock14 = Resblock()
        self.resblock15 = Resblock()
        self.flatten1 = nn.Flatten()
        self.fc_mu = nn.Linear(hidden_dim * seq_len, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * seq_len, latent_dim)
        
        # Build Decoder
        self.linear2 = nn.Linear(latent_dim, hidden_dim * seq_len)
        self.reshape2 = Reshape(hidden_dim, seq_len)
        self.resblock21 = Resblock()
        self.resblock22 = Resblock()
        self.resblock23 = Resblock()
        self.resblock24 = Resblock()
        self.resblock25 = Resblock()
        self.conv2 = nn.Conv1d(in_channels=self.motif_conv_hidden, out_channels=4, kernel_size=self.conv_width_motif, padding='same')
        
    def EncoderNet(self, x):
        x = self.conv1(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.resblock15(x)
        x = self.flatten1(x)
        z_mean = self.fc_mu(x)
        z_logvar = self.fc_var(x)
        return z_mean, z_logvar
    
    def DecoderNet(self, z):
        z = self.linear2(z)
        z = self.reshape2(z)
        z = self.resblock21(z)
        z = self.resblock22(z)
        z = self.resblock23(z)
        z = self.resblock24(z)
        z = self.resblock25(z)
        x = self.conv2(z)
        x = F.softmax(x, dim=1)
        return x
    
    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return eps * std + z_mean
    
    def forward(self, x):
        z_mean, z_logvar = self.EncoderNet(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.DecoderNet(z)
        
        return  [x_recon, x, z_mean, z_logvar, z]
    
    def log_normal_pdf(self, z, mean, logvar):
        mean = torch.tensor(mean)
        logvar = torch.tensor(logvar)
        log2pi = torch.log(2. * torch.tensor(np.pi))
        res = -0.5 * torch.sum( torch.mul((z-mean)**2, torch.exp(-logvar)) + logvar + log2pi )
        return res
    
    def compute_loss(self, args):
        x_recon = args[0]
        x_real = args[1]
        z_mean = args[2]
        z_logvar = args[3]
        z = args[4]
        
        logpx_z = torch.sum(torch.log( torch.sum( torch.multiply(x_recon, x_real) , axis=1) ))
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, z_mean, z_logvar)
        loss = - torch.sum(logpx_z + logpz - logqz_x)
        return loss
    
    def sample(self, sample_number):
        fake_z = torch.randn(sample_number, self.latent_dim).to(self.device)
        fake_x = self.DecoderNet(fake_z)
        fake_x = onehot2seq(fake_x)
        return fake_x
    


class SimpleVAE:
    
    def __init__(self,
                 n_tokens = 4,
                 latent_dim = 128,
                 batch_size = 32,
                 num_epochs = 1000,
                 save_epoch = 10,
                 print_epoch = 10,
                 length = 80,
                 model_name = "vae",
                 lr = 1e-5
                 ):
    
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_epoch = save_epoch
        self.print_epoch = print_epoch
        self.seq_len = length
        self.model_name = model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = VAE(seq_len = length).to(self.device)
        self.learning_rate = lr
    
    def make_dir(self):
        self.base_dir = os.path.join(self.checkpoint_root, self.model_name)
        print("results will be saved in: ", self.base_dir)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        if not os.path.exists(self.base_dir + '/training_log'):
            os.makedirs(self.base_dir + '/training_log')

        if not os.path.exists(self.base_dir + '/checkpoints'):
            os.makedirs(self.base_dir + '/checkpoints')

    
    def train(self, dataset = None, savepath = None):
        
        self.file = dataset
        self.checkpoint_root = savepath
        
        self.make_dir()
        train_dataset = Seq_loader(self.file, seq_len=self.seq_len)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size, shuffle=True)

        vae = self.vae
        optimizer = optim.Adam(vae.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        
        ## Training precedure
        for epoch in range(self.num_epochs):
            begin_time = time()
            loss_sum = []
            for i, sample in enumerate(train_loader):
                real_samples = sample['onehot'].to(self.device).float()
                x = real_samples.permute(0,2,1)
                res = vae.forward(x)
                loss = vae.compute_loss(res)
                
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5, norm_type=2)
                optimizer.step()
                
                loss_sum.append(loss.detach().tolist())
            
            loss_sum = np.mean(loss_sum)
            training_time = time() - begin_time
            minute = int(training_time // 60)
            second = int(training_time % 60)
            print(f'epoch {epoch}: training loss: {loss_sum}, time cost: {minute} m:{second} s')
            if (epoch + 1) % self.print_epoch == 0:
                samples = vae.sample(100)
                sequence2fa(samples, self.base_dir + '/training_log' + '/gen_' + 'iter_{}.txt'.format(epoch+1))
            if (epoch + 1) % self.save_epoch == 0:
                torch.save(vae.state_dict(), self.base_dir + '/checkpoints' + '/vae.pth') # '/vae_' + str(epoch+1) + '.pth'
    
    def generate(self, sample_model_path=None, sample_number=None, sample_output = True, seed = 0):
        self.sample_model_path = sample_model_path
        self.sample_number = sample_number
        self.sample_output = sample_output
        self.sample_model_dir = os.path.dirname(self.sample_model_path)
        torch.manual_seed(seed)

        model = self.vae
        checkpoint = torch.load(self.sample_model_path)
        model.load_state_dict(checkpoint)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model = model.eval()
        
        sample_text = model.sample(sample_number)
        
        if (self.sample_output):
            path_samples = os.path.join(self.sample_model_dir, 'samples/sample_{}.txt'.format(seed))
            if not os.path.exists(os.path.dirname(path_samples)):
                os.mkdir(os.path.dirname(path_samples))
            print("samples will be saved in: ", path_samples)
            with open(path_samples, 'w') as f:
                for i,item in enumerate(sample_text):
                    f.write('>' + str(i) + '\n')
                    f.write(item + '\n')
        
        return sample_text
    
    def embedding(self, sample_model_path=None, seqs=None):
        self.sample_model_path = sample_model_path
        self.sample_model_dir = os.path.dirname(self.sample_model_path)

        model = self.vae
        checkpoint = torch.load(self.sample_model_path)
        model.load_state_dict(checkpoint)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model = model.eval()

        x = seq2onehot(seqs)
        x = torch.tensor(x)
        x = x.permute(0,2,1).to(device).float()
        
        z_mean, z_logvar = model.EncoderNet(x)
        z = model.reparameterize(z_mean, z_logvar)
        
        return z.tolist()
