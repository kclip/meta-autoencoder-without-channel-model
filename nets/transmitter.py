from __future__ import print_function
import torch
import torch.nn as nn


class transmitter(nn.Module):
    def __init__(self, M, num_neurons_encoder, n, if_bias, if_relu):
        super(transmitter, self).__init__()
        self.enc_fc1 = nn.Linear(M, num_neurons_encoder, bias=if_bias)
        self.enc_fc2 = nn.Linear(num_neurons_encoder, n, bias=if_bias)

        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()

    def forward(self, x, device, relax_sigma, Noise_relax):
        x = self.enc_fc1(x)
        x = self.activ(x)
        x = self.enc_fc2(x)
        # normalize
        x_norm = torch.norm(x, dim=1)
        x_norm = x_norm.unsqueeze(1)
        x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm  # since each has ^2 norm as 0.5 -> complex 1

        if relax_sigma > 0:
            x = pow(1 - pow(relax_sigma, 2), 0.5) * x # this will be the actual mean
            w = torch.zeros(x.shape[0], x.shape[1])
            for relax_batch_ind in range(x.shape[0]):
                w[relax_batch_ind] = Noise_relax.sample()
            w = w.type(torch.FloatTensor).to(device)
            x_rel = (x + w).clone().detach() # relaxation, this is one realization
        else:
            x_rel = x.clone().detach()

        # x: for update transmitter, x_rel: actual transmitted symbol which is merely a value (realization)
        return x, x_rel


def tx_dnn(**kwargs):
    tx_net = transmitter(**kwargs)
    return tx_net


