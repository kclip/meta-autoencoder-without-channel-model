from __future__ import print_function
import torch.nn as nn
from torch.nn import functional as F
from utils.basic_funcs import complex_conv_transpose

# in this way of defining receiver neural network, we can consider weights which are function of other weights

class meta_receiver(nn.Module):
    def __init__(self, if_relu): # it only gets paramters from other network's parameters
        super(meta_receiver, self).__init__()
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()

    def forward(self, x, var, if_bias, device, if_RTN):
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
                # RTN
                if if_RTN:
                    if if_bias:
                        w_rtn_1, b_rtn_1 = var[idx], var[idx+1]
                        h_inv = F.linear(x, w_rtn_1, b_rtn_1)
                        h_inv = self.tanh(h_inv)
                        w_rtn_2, b_rtn_2 = var[idx+2], var[idx + 3]
                        h_inv = F.linear(h_inv, w_rtn_2, b_rtn_2)
                        h_inv = self.tanh(h_inv)
                        w_rtn_3, b_rtn_3 = var[idx + 4], var[idx + 5]
                        h_inv = F.linear(h_inv, w_rtn_3, b_rtn_3)
                        rtn_gap = 6
                    else:
                        w_rtn_1 = var[idx]
                        h_inv = F.linear(x, w_rtn_1)
                        h_inv = self.tanh(h_inv)
                        w_rtn_2 = var[idx+1]
                        h_inv = F.linear(h_inv, w_rtn_2)
                        h_inv = self.tanh(h_inv)
                        w_rtn_3 = var[idx+2]
                        h_inv = F.linear(h_inv, w_rtn_3)
                        rtn_gap = 3
                    x = complex_conv_transpose(h_inv, x)
                    x = x.to(device)
                else:
                    rtn_gap = 0
                # from now on, decoder
                if if_bias:
                    w3, b3 = var[idx+ rtn_gap], var[idx + rtn_gap + 1]  # weight and bias
                    x = F.linear(x, w3, b3)
                    idx += (2 + rtn_gap)
                else:
                    w3 = var[idx + rtn_gap]  # weight
                    x = F.linear(x, w3)
                    idx += (1 + rtn_gap)
            elif idx == gap + rtn_gap+idx_init:
                if if_bias:
                    w4, b4 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w4, b4)
                    idx += 2
                else:
                    w4 = var[idx]  # weight
                    x = F.linear(x, w4)
                    idx += 1
        return x

def meta_rx(**kwargs):
    net = meta_receiver(**kwargs)
    return net
