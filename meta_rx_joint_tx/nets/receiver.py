from __future__ import print_function
import torch.nn as nn
from utils.basic_funcs import complex_conv_transpose, reset_randomness

class receiver(nn.Module):
    def __init__(self, M, n, n_inv_filter, num_neurons_decoder, if_bias, if_relu, if_RTN, if_fix_random_seed, random_seed):
        super(receiver, self).__init__()
        num_inv_filter = 2 * n_inv_filter
        if if_RTN:
            if if_fix_random_seed:
                reset_randomness(random_seed+1)
            self.rtn_1 = nn.Linear(n, n, bias=if_bias)
            self.rtn_2 = nn.Linear(n, n, bias=if_bias)
            self.rtn_3 = nn.Linear(n, num_inv_filter, bias=if_bias)
        else:
            pass
        if if_fix_random_seed:
            reset_randomness(random_seed)
        self.dec_fc1 = nn.Linear(n, num_neurons_decoder, bias=if_bias)
        self.dec_fc2 = nn.Linear(num_neurons_decoder, M, bias=if_bias)
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()
    def forward(self, x, if_RTN, device): # this x is the received signal
        #### RTN
        if if_RTN:
            h_inv = self.rtn_1(x)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_2(h_inv)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_3(h_inv)  # no activation for the final rtn (linear activation without weights)
            x = complex_conv_transpose(h_inv, x)
            x = x.to(device)
        else:
            pass
        x = self.dec_fc1(x)
        x = self.activ(x)
        x = self.dec_fc2(x)  # softmax taken at loss function

        return x

def rx_dnn(**kwargs):
    rx_net = receiver(**kwargs)
    return rx_net

