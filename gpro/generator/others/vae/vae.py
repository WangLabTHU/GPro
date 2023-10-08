import os,sys
import numpy as np
from tqdm import tqdm
from time import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from gpro.generator.wgan.common import Seq_loader, onehot2seq, sequence2fa

'''
input should be [batch_size, seq_len, n_tokens]
'''

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)


class VAE(nn.Module):
    def __init__(self, n_tokens = 4, latent_dim = 128, seq_len = 50):
        super(VAE, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build Encoder

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_tokens * seq_len, latent_dim, bias=True),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        
        # Build Decoder
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Linear(latent_dim, n_tokens * seq_len, bias=True),
            nn.ReLU(),
            Reshape(seq_len, n_tokens),
            nn.Softmax(dim=2)
        )
        
    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
        
    def decode(self, z):    
        result = self.decoder(z)
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), x, mu, log_var]
    
    def generate(self, x):
        return self.forward(x)[0]
    
    def sample(self, sample_number):
        fake_z = torch.randn(sample_number, self.latent_dim).to(self.device)
        fake_x = self.decode(fake_z)
        fake_x = onehot2seq(fake_x)
        return fake_x
    


class SimpleVAE:
    
    def __init__(self,
                 n_tokens = 4,
                 latent_dim = 128,
                 batch_size = 32,
                 num_epochs = 1000,
                 save_epoch = 100,
                 print_epoch = 100,
                 length = 50,
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
    
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    """
    def loss_function(self, *args, **kwargs):
        recons = args[0]
        origin = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, origin)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    
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
                result = vae(real_samples)
                loss = self.loss_function(*result, M_N = 0.00025)["loss"]
                optimizer.zero_grad()
                loss.backward()
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
                torch.save(vae.state_dict(), self.base_dir + '/checkpoints' + '/vae_' + str(epoch+1) + '.pth')
                

if __name__ == "__main__":
    # y = torch.rand(size=[2,4])
    # print( nn.Softmax(dim=1)(y) )
    
    dataset_path = '/home/qxdu/gpro/gpro/gpro/data/diffusion_promoter/sequence_data.txt'
    checkpoint_path = '/home/qxdu/gpro/gpro/gpro/checkpoints'
    model = SimpleVAE(length=50)
    model.train(dataset=dataset_path, savepath=checkpoint_path)
