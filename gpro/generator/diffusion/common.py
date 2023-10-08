import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from .datasets.dataset_promoter import PromoterDataset
from .diffusion_utils.diffusion_multinomial import MultinomialDiffusion
from .layers.transformer import LinearAttentionTransformerEmbedding

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler

## Part 1 Data Part

def get_data(dataset, dataset_type, validation, batch_size, num_workers, length):
    
    dataset_choices = {'promoter', 'padding', 'amino_acid'}
    assert dataset_type in dataset_choices
    
    if dataset_type == 'promoter':
        train = PromoterDataset(file=dataset, seq_len=length, split='train')
        valid = PromoterDataset(file=dataset, seq_len=length, split='valid')
        test =  PromoterDataset(file=dataset, seq_len=length, split='test')
        data_shape = (length,)
        num_classes = 4

    if dataset_type == 'padding':
        train = PromoterDataset(file=dataset, seq_len=length, split='train')
        valid = PromoterDataset(file=dataset, seq_len=length, split='valid')
        test =  PromoterDataset(file=dataset, seq_len=length, split='test')
        data_shape = (length,)
        num_classes = 5
    
    if dataset_type == 'amino_acid':
        train = PromoterDataset(file=dataset, seq_len=length, split='train')
        valid = PromoterDataset(file=dataset, seq_len=length, split='valid')
        test =  PromoterDataset(file=dataset, seq_len=length, split='test')
        data_shape = (length,)
        num_classes = 20
    
    # Data Loader
    if validation:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        eval_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    else:
        dataset_train = ConcatDataset([train, valid])
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        eval_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, eval_loader, data_shape, num_classes


## Part 2 Model Part

class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x

def get_model(num_classes, data_shape, transformer_dim, transformer_heads, transformer_depth, transformer_blocks, 
              transformer_local_heads, transformer_local_size, diffusion_steps):
    data_shape = data_shape
    num_classes = num_classes
    transformer_dim = transformer_dim
    transformer_heads = transformer_heads
    transformer_depth = transformer_depth
    transformer_blocks = transformer_blocks
    transformer_local_heads = transformer_local_heads
    transformer_local_size = transformer_local_size
    diffusion_steps = diffusion_steps
    diffusion_loss = 'vb_stochastic'
    diffusion_parametrization = 'x0'

    C, L = 1, data_shape[0]

    current_shape = (L,)

    class DynamicsTransformer(nn.Module):
        def __init__(self):
            super(DynamicsTransformer, self).__init__()
            self.transformer = LinearAttentionTransformerEmbedding(
                input_dim=num_classes,
                output_dim=num_classes,
                dim=transformer_dim,
                heads=transformer_heads,
                depth=transformer_depth,
                n_blocks=transformer_blocks,
                max_seq_len=L,
                num_timesteps=diffusion_steps,
                causal=False,
                ff_dropout=0,
                attn_dropout=0,
                n_local_attn_heads=transformer_local_heads,
                local_attn_window_size=transformer_local_size,
                reversible=False,
            )

            self.rezero = Rezero()

        def forward(self, t, x):
            x = self.transformer(x, t)
            x = x.permute(0, 2, 1)
            x = self.rezero(x)
            return x

    dynamics = DynamicsTransformer()

    base_dist = MultinomialDiffusion(
        num_classes, current_shape, dynamics,
        timesteps=diffusion_steps,
        loss_type=diffusion_loss,
        parametrization=diffusion_parametrization)
    return base_dist



## Part3 Optimizer Part
class LinearWarmupScheduler(_LRScheduler):
    """ Linearly warm-up (increasing) learning rate, starting from zero.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_epoch: target learning rate is reached at total_epoch.
    """

    def __init__(self, optimizer, total_epoch, last_epoch=-1):
        self.total_epoch = total_epoch
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(1, (self.last_epoch / self.total_epoch)) for base_lr in self.base_lrs]


def get_optim(optimizer, lr, warmup, momentum, momentum_sqr, gamma, model):
    
    optim_choices = {'sgd', 'adam', 'adamax'}
    assert optimizer in optim_choices

    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(momentum, momentum_sqr))
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, betas=(momentum, momentum_sqr))

    if warmup is not None:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=warmup)
    else:
        scheduler_iter = None

    scheduler_epoch = ExponentialLR(optimizer, gamma=gamma)
    
    return optimizer, scheduler_iter, scheduler_epoch
