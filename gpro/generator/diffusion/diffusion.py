import os
import sys

import math
import torch
import time
import pickle

from torch.utils.tensorboard import SummaryWriter
from .diffusion_utils.experiment import BaseExperiment
from .diffusion_utils.utils import clean_dict, get_args_table
from .common import get_data, get_model, get_optim


class DiffusionExperiment(BaseExperiment):
    no_log_keys = ['project', 'name',
                   'log_tb', 'check_every', 'eval_every',
                   'device', 'parallel'
                   'pin_memory', 'num_workers']

    def __init__(self,
                 # batch_size = 32,
                 batch_size = 64,
                 update_freq = 1,
                 lr = 1e-4,
                 epochs = 200,
                 eval_every = 2,
                 check_every = 20,
                 dataset_type = "promoter",
                 length = 50,
                 # diffusion_steps = 1000,
                 diffusion_steps = 100,
                 transformer_depth = 12,
                 transformer_heads = 16,
                 transformer_local_heads = 8,
                 transformer_local_size = 25,
                 gamma = 0.99,
                 model_name = "diffusion",
                 
                 dataset = None, # Flexiable parts
                 savepath = None,
                 sample_model_path = None, 
                 sample_number = None,
                 sample_output = None
                 ):
        
        
        ## Part 1: experiment parameters
        self.epochs = epochs
        self.seed = 0
        self.device = 'cuda'
        self.name = model_name
        self.project = None
        self.eval_every = eval_every
        self.check_every = check_every
        self.log_tb = True # Tensorboard Results
        self.resume = None
        
        ## Part2: data parameters
        self.dataset_type = dataset_type
        self.validation = True
        self.batch_size = batch_size
        self.seq_len = length
        self.num_workers = 4
        
        ## Part3: Model parameters
        self.transformer_dim = 512
        self.transformer_heads  = transformer_heads
        self.transformer_depth = transformer_depth
        self.transformer_blocks = 1
        self.transformer_local_heads = transformer_local_heads
        self.transformer_local_size =  transformer_local_size
        self.diffusion_steps = diffusion_steps
        
        
        ## Part4: Optimization parameters
        self.optimizer_mode = 'adam'
        self.lr = lr
        self.warmup = None
        self.update_freq = update_freq # resaved
        self.momentum = 0.9
        self.momentum_sqr = 0.999
        self.gamma = gamma
        self.debug = 0

        ## Flexiable parts
        self.dataset = dataset # input dataset 
        self.log_base = savepath # result savepath
        self.sample_model_path = sample_model_path
        self.sample_number = sample_number
        self.sample_output = sample_output
    
    def log_fn(self, epoch, train_dict, eval_dict):

        # Tensorboard
        if self.log_tb:
            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)


    def resume(self):
        resume_path = os.path.join(self.log_base, self.resume, 'check')
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]
            if epoch in self.eval_epochs:
                eval_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    eval_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else: eval_dict = None
            self.log_fn(epoch, train_dict=train_dict, eval_dict=eval_dict)


    def clear_existed_dataset(self, dataset):
        file_list = dataset.split("/")
        file_list.pop()
        root = '/'.join(file_list)
        
        print("clearing existed dataset in ", root)
        
        vocab_file = os.path.join(root, 'vocab.json')
        if os.path.exists(vocab_file):
            os.remove(vocab_file)
            print("removing: ", vocab_file)
        
        for split in {'train', 'valid', 'test'}:
            split_file = os.path.join(root, 'processed_{}.pt'.format(split))
            if os.path.exists(split_file):
                os.remove(split_file)
                print("removing: ", split_file)
    
    def train(self, dataset = None, savepath = None):
        
        if (self.dataset == None):
            self.dataset = dataset
        if (self.log_base == None):
            self.log_base = savepath
        
        # Part 1 Preparation
        self.clear_existed_dataset(self.dataset)
        
        # Part 2 derivation
        self.train_loader, self.eval_loader, self.data_shape, self.num_classes = get_data(self.dataset, self.dataset_type, self.validation, 
                                                                                          self.batch_size, self.num_workers, self.seq_len)
        self.data_id = self.dataset_type
        
        # Part 3 derivation
        self.model = get_model(self.num_classes, self.data_shape, self.transformer_dim, self.transformer_heads, self.transformer_depth, self.transformer_blocks, 
                          self.transformer_local_heads, self.transformer_local_size, self.diffusion_steps)
        self.model_id = 'multinomial_diffusion_v2'
        
        # Part 4 derivation
        self.optimizer, self.scheduler_iter, self.scheduler_epoch = get_optim(self.optimizer_mode, self.lr, self.warmup, self.momentum, 
                                                                              self.momentum_sqr, self.gamma, self.model)
        self.optim_id = "expdecay"

        # Edit args
        if self.eval_every is None:
            self.eval_every = self.epochs
        if self.check_every is None:
            self.check_every = self.epochs
        if self.project is None:
            self.project = '_'.join([self.data_id, self.model_id])

        # Move model
        self.model = self.model.to(self.device)

        # Init parent
        super(DiffusionExperiment, self).__init__(model=self.model,
                                                  optimizer=self.optimizer,
                                                  scheduler_iter=self.scheduler_iter,
                                                  scheduler_epoch=self.scheduler_epoch,
                                                  log_path=os.path.join(self.log_base, self.name),
                                                  eval_every=self.eval_every,
                                                  check_every=self.check_every)        
        
        args_dict = {"dataset": self.dataset, "savepath": self.log_base, "batch_size": self.batch_size,
                     "learning rate": self.lr, "epochs": self.epochs, "steps": self.diffusion_steps,
                     "dataset_type": self.dataset_type, "validation": self.validation, 
                     "num_workers": self.num_workers, "length": self.seq_len, 
                     "transformer_dim": self.transformer_dim, "transformer_heads": self.transformer_heads,
                     "transformer_depth": self.transformer_depth, "transformer_blocks": self.transformer_blocks, 
                     "transformer_local_heads": self.transformer_local_heads,
                     "transformer_local_size": self.transformer_local_size, "diffusion_steps": self.diffusion_steps}
                
        # Store args
        self.create_folders()
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args_dict, f)
        print("args have been storage")
        
        # Init logging
        if self.log_tb:
            self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
            self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)
        
        if self.resume: self.resume()
        super(DiffusionExperiment, self).run(epochs=self.epochs)
    
    def generate(self, sample_model_path=None, sample_number=None, sample_output = True, seed = 0):
        
        if (self.sample_model_path == None):
            self.sample_model_path = sample_model_path
        if (self.sample_number == None):
            self.sample_number = sample_number
        if (self.sample_output == None):
            self.sample_output = sample_output
        
        self.sample_model_dir = os.path.dirname(os.path.dirname(self.sample_model_path))
        path_args = '{}/args.pickle'.format(self.sample_model_dir)
        path_check = '{}/check/checkpoint.pt'.format(self.sample_model_dir)
        
        torch.manual_seed(seed)
        with open(path_args, 'rb') as f:
            args = pickle.load(f)
        print(args)
        length = int(args["length"])
        
        train_loader, eval_loader, data_shape, num_classes = get_data(args["dataset"], args["dataset_type"], args["validation"], 
                                                              args["batch_size"], args["num_workers"], args["length"])

        model = get_model(num_classes, data_shape, args["transformer_dim"], args["transformer_heads"], args["transformer_depth"], args["transformer_blocks"], 
                          args["transformer_local_heads"], args["transformer_local_size"], args["diffusion_steps"])

        checkpoint = torch.load(path_check)
        model.load_state_dict(checkpoint['model'])
        print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args["epochs"]))


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model = model.eval()
        lengths = torch.ones(self.sample_number, device=device, dtype=torch.long) * length
        
        samples_chain = model.sample_chain(self.sample_number)
        samples = samples_chain[0]
        samples_text = train_loader.dataset.vocab.decode(samples.cpu(), lengths.cpu())
        print([len(s) for s in samples_text])
        
        if (self.sample_output):
            path_samples = os.path.join(self.sample_model_dir, 'samples/sample_ep{}_s{}_num_{}.txt'.format(checkpoint['current_epoch'], seed, sample_number))
            if not os.path.exists(os.path.dirname(path_samples)):
                os.mkdir(os.path.dirname(path_samples))
            print("samples will be saved in: ", path_samples)
            with open(path_samples, 'w') as f:
                for i,item in enumerate(samples_text):
                    f.write('>' + str(i) + '\n')
                    f.write(item + '\n')
        
        return samples_text
        
            

class Diffusion_language(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        loss_moving = None
        for iteration, (x, length) in enumerate(self.train_loader):
            x, length = x.to(self.device), length.to(self.device)
            num_elem = length.sum()
            loss = - self.model.log_prob(x).sum() / (math.log(2) * num_elem)
            loss.backward()
            if (iteration + 1) % self.update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)

            if loss_moving is None:
                loss_moving = loss.detach().cpu().item()
            else:
                loss_moving = .99 * loss_moving + .01 * loss.detach().cpu().item()

            if self.debug and loss_count > self.debug:
                break
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, self.epochs, loss_count, len(self.train_loader.dataset), loss_moving), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpc': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()
        sqrt_Lt = torch.sqrt(self.model.Lt_history)
        
        # print('sqrt |Lt_history|^2')
        # print(' '.join(f'{item.item():.2f}' for item in sqrt_Lt))
        # print()
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x, length in self.eval_loader:
                x, length = x.to(self.device), length.to(self.device)
                num_elem = length.sum()
                loss = - self.model.log_prob(x).sum() / (math.log(2) * num_elem)
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating train. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, self.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x, length in self.eval_loader:
                x, length = x.to(self.device), length.to(self.device)
                num_elem = length.sum()
                loss = - self.model.log_prob(x).sum() / (math.log(2) * num_elem)
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, self.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        return {'bpc': loss_sum/loss_count}
    
