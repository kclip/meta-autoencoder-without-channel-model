from __future__ import print_function
import torch.nn as nn
import torch
from torch.nn import functional as F

class meta_transmitter(nn.Module):
    def __init__(self, if_relu): # it only gets paramters from other network's parameters
        super(meta_transmitter, self).__init__()
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()

    def forward(self, x, var, if_bias, device, relax_sigma, Noise_relax):
        idx_init = 0
        if if_bias:
            gap = 2
        else:
            gap = 1
        idx = idx_init
        while idx < len(var):
            if idx > idx_init: # no activation from the beginning
                x = self.activ(x)
            if idx == idx_init:
                if if_bias:
                    w1, b1 = var[idx], var[idx + 1] # weight and bias
                    x = F.linear(x, w1, b1)
                    idx += 2
                else:
                    w1 = var[idx] # weight
                    x = F.linear(x, w1)
                    idx += 1
            elif idx == gap * 1+idx_init:
                if if_bias:
                    w2, b2 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w2, b2)
                    idx += 2
                else:
                    w2 = var[idx]  # weight and bias
                    x = F.linear(x, w2)
                    idx += 1
        # normalize
        x_norm = torch.norm(x, dim=1)
        x_norm = x_norm.unsqueeze(1)
        x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm
        if relax_sigma > 0:
            x = pow(1 - pow(relax_sigma, 2), 0.5) * x  # this will be the actual mean
            w = torch.zeros(x.shape[0], x.shape[1])
            for relax_batch_ind in range(x.shape[0]):
                w[relax_batch_ind] = Noise_relax.sample()
            w = w.type(torch.FloatTensor).to(device)
            x_rel = (x + w).clone().detach()  # relaxation, this is one realization
        else:
            x_rel = x.clone().detach()
        # x: for update transmitter, x_rel: actual transmitted symbol which is merely a value (realization)
        return x, x_rel

def meta_tx(**kwargs):
    net = meta_transmitter(**kwargs)
    return net
