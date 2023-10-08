import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from .common import Resblock, Seq_loader, onehot2seq, sequence2fa
from torch.utils.data import DataLoader


class Generator(nn.Module):

    def __init__(self, model_dim = 512, resblock_kernelsize = 5, onehot_size = 4, nz = 128, seq_len = 50):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.linear = nn.Linear(nz, model_dim * seq_len, bias=True)
        self.resblock1 = Resblock(kernel_size=resblock_kernelsize)
        self.resblock2 = Resblock(kernel_size=resblock_kernelsize)
        self.resblock3 = Resblock(kernel_size=resblock_kernelsize)
        self.resblock4 = Resblock(kernel_size=resblock_kernelsize)
        self.resblock5 = Resblock(kernel_size=resblock_kernelsize)
        self.conv = nn.Conv1d(model_dim, onehot_size, 1)

    def forward(self, gen_inputs):
        outputs = self.linear(gen_inputs)
        outputs = outputs.view(-1, self.model_dim, self.seq_len)
        outputs = self.resblock1(outputs)
        outputs = self.resblock2(outputs)
        outputs = self.resblock3(outputs)
        outputs = self.resblock4(outputs)
        outputs = self.resblock5(outputs)
        outputs = self.conv(outputs)
        outputs = nn.Softmax(dim=1)(outputs)
        return outputs

class Discriminator(nn.Module):

    def __init__(self, model_dim = 512, resblock_kernelsize = 5, seq_len = 50):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.conv = nn.Conv1d(4, model_dim, kernel_size=5, padding=2)
        self.resblock1 = Resblock(kernel_size=resblock_kernelsize)  # For visualize results by torchinfo
        self.resblock2 = Resblock(kernel_size=resblock_kernelsize)
        self.resblock3 = Resblock(kernel_size=resblock_kernelsize)
        self.resblock4 = Resblock(kernel_size=resblock_kernelsize)
        self.resblock5 = Resblock(kernel_size=resblock_kernelsize)
        self.linear = nn.Linear(seq_len * model_dim, 1, bias=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.resblock1(outputs)
        outputs = self.resblock2(outputs)
        outputs = self.resblock3(outputs)
        outputs = self.resblock4(outputs)
        outputs = self.resblock5(outputs)
        # outputs = self.resblock_part(outputs) # For visualize results by torchinfo
        outputs = outputs.reshape(-1, self.seq_len * self.model_dim)
        outputs = self.linear(outputs)
        return outputs

def cal_gradient_penalty(netD, real_samples, fake_samples):
    # Calculate the loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = torch.randn((len(fake_samples), 1, 1)).to(device)    
    differences = fake_samples - real_samples
    interpolates = (real_samples + alpha * differences).requires_grad_(True)

    # Calculate the penalty cost
    out_interpolates = netD(interpolates)
    weight = torch.ones(out_interpolates.size()).to(device)
    gradients = torch.autograd.grad(outputs=out_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=weight,
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradients_l2norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
    gradients_gp = torch.mean((gradients_l2norm - 1) ** 2)
    return gradients_gp

class WGAN_language:
    
    def __init__(self, 
                 nz = 128,
                 netG_lr = 1e-4,
                 netD_lr = 1e-4,
                 batch_size = 32,
                 num_epochs = 12,
                 print_epoch = 1,
                 save_epoch = 1,
                 Lambda = 10,
                 length = 50,
                 model_name = 'wgan',
                 
                 dataset = None, # Flexiable parts
                 savepath = None,
                 sample_model_path = None, 
                 sample_number = None,
                 sample_output = None
                 ):
        
        self.nz = nz
        self.netG_lr = netG_lr
        self.netD_lr = netD_lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_epoch = print_epoch
        self.save_epoch = save_epoch
        self.LAMBDA = Lambda
        self.model_name = model_name
        self.seq_len = length
        
        # FIXED PART
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG = Generator(seq_len=length).to(self.device)
        self.netD = Discriminator(seq_len=length).to(self.device)
        
        # FLEXIABLE PART
        self.file = dataset
        self.checkpoint_root = savepath
        self.sample_model_path = sample_model_path
        self.sample_number = sample_number
        self.sample_output = sample_output

    def make_dir(self):
        self.base_dir = os.path.join(self.checkpoint_root, self.model_name)
        print("results will be saved in: ", self.base_dir)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        if not os.path.exists(self.base_dir + '/figure'):
            os.makedirs(self.base_dir + '/figure')

        if not os.path.exists(self.base_dir + '/training_log'):
            os.makedirs(self.base_dir + '/training_log')

        if not os.path.exists(self.base_dir + '/checkpoints'):
            os.makedirs(self.base_dir + '/checkpoints')

    def train(self, dataset = None, savepath = None):
        
        if (self.file == None):
            self.file = dataset
        if (self.checkpoint_root == None):
            self.checkpoint_root = savepath
        
        self.make_dir()
        train_dataset = Seq_loader(self.file, seq_len=self.seq_len)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size, shuffle=True)

        # Define the neural networks
        netG = self.netG
        netD = self.netD
        # define the optimizer
        optimizer_G = optim.Adam(netG.parameters(), lr=self.netG_lr, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(netD.parameters(), lr=self.netD_lr, betas=(0.5, 0.9))
        ## Training precedure
        Loss = {}
        for epoch in range(self.num_epochs):
            for i, sample in enumerate(train_loader):
                ## Train the discriminator
                # Forward pass
                for p in netD.parameters():
                    p.requires_grad = True  # optimize D
                netD.zero_grad()  # initialize the grad
                real_samples = sample['onehot'].permute(0, 2, 1).to(self.device).float()
                real_samples = torch.autograd.Variable(real_samples)
                rand_tensor = torch.randn((len(real_samples), self.nz)).requires_grad_(True).to(self.device)
                # rand_tensor = torch.autograd.Variable(rand_tensor)
                fake_samples = netG(rand_tensor)
                fake_samples = fake_samples.detach()  # Detach not optimize the generator
                real_output = netD(real_samples)
                fake_output = netD(fake_samples)

                # WGAN loss with gradient panelty
                gradients_penalty = cal_gradient_penalty(netD, real_samples, fake_samples)
                # gradients_penalty = calc_gradient_penalty(netD,real_samples,fake_samples)
                disc_cost = fake_output.mean() - real_output.mean()
                disc_cost = disc_cost + self.LAMBDA * gradients_penalty
                # Optimize discriminator
                # disc_cost.backward(retain_graph=True)
                disc_cost.backward()
                optimizer_D.step()

                ## Train the generator
                for p in netD.parameters():
                    p.requires_grad = False  # freeze D
                netG.zero_grad()
                rand_tensor = torch.randn((len(fake_samples), self.nz)).to(self.device)
                # rand_tensor = torch.autograd.Variable(rand_tensor)
                fake_samples = netG(rand_tensor)
                fake_output = netD(fake_samples)
                gen_cost = -fake_output.mean()
                # Optimize generator
                gen_cost.backward()
                optimizer_G.step()

                ## Print the training loss
                if i % 3 == 0:
                    print("epoch[{}/{}] iter[{}/{}]: Training loss: disc_loss: {}, gradients_gp: {}, gen_loss: {}".
                          format(epoch + 1, self.num_epochs, i, len(train_loader), disc_cost, gradients_penalty, gen_cost))
                    # print(fake_samples[0][:,0:10])

                # print the training results
            if epoch % self.print_epoch == 0:
                netG.eval()
                all_sequence = []
                for m in range(50):
                    # torch.manual_seed(m+10086)  # Set one determined random seed
                    fixed_noise = torch.randn((self.batch_size, self.nz)).to(self.device)  #
                    fake_samples = netG(fixed_noise)  # Generate samples
                    fake_samples = fake_samples.permute(0, 2, 1)
                    sequence = onehot2seq(fake_samples, self.seq_len)  # Get the sequence and save
                    all_sequence = all_sequence + sequence
                sequence2fa(all_sequence, self.base_dir + '/training_log' + '/gen_' + 'iter_{}.txt'.format(epoch))  # Save the file

            if (epoch + 1) % self.save_epoch == 0:
                torch.save({'model': netG.state_dict()}, self.base_dir + '/checkpoints' + '/net_G_' + str(epoch+1) + '.pth')
                torch.save({'model': netD.state_dict()}, self.base_dir + '/checkpoints' + '/net_D_' + str(epoch+1) + '.pth')

    def generate(self, sample_model_path=None, sample_number=None, sample_output = True, seed = 0):
        
        if (self.sample_model_path == None):
            self.sample_model_path = sample_model_path
        if (self.sample_number == None):
            self.sample_number = sample_number
        if (self.sample_output == None):
            self.sample_output = sample_output
        
        tmp = os.path.basename(self.sample_model_path).split(".")[0]
        cur_epoch = tmp.split("_")[-1]
        
        torch.manual_seed(seed)
        if (self.sample_number > 10000):
            print('Too many numbers, it must < 10000')
            return 0
        
        # load and evaluate model:
        netG = self.netG
        state_dict = torch.load(self.sample_model_path)
        netG.load_state_dict(state_dict['model']) # ['model']
        netG.eval()
        
        file_list = self.sample_model_path.split("/")
        file_list.pop()
        file_list.pop()
        root = '/'.join(file_list)
        
        with torch.no_grad():
            fixed_noise = torch.randn((self.sample_number, self.nz)).to(self.device)
            out_data = netG(fixed_noise)
            out_data = out_data.permute(0, 2, 1)
            all_sequence = onehot2seq(out_data, seq_len=self.seq_len)
            
        if (self.sample_output):
            # path_samples = os.path.join(root, 'samples/sample_{}.txt'.format(seed))
            path_samples = os.path.join(root, 'samples/sample_ep{}_s{}_num_{}.txt'.format(cur_epoch, seed, sample_number))
            if not os.path.exists(os.path.dirname(path_samples)):
                os.mkdir(os.path.dirname(path_samples))
            print("samples will be saved in: ", path_samples)
            sequence2fa(all_sequence, path_samples)
        return all_sequence



    








