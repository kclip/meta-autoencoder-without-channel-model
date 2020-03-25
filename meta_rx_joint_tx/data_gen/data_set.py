import torch

def message_gen(k, mb_size):
    tot_message_num = pow(2,k)
    m = torch.zeros(mb_size, tot_message_num)
    label = torch.zeros(mb_size)
    for ind_mb in range(mb_size):
        if ind_mb % tot_message_num == 0:
            rand_lst = torch.randperm(tot_message_num)
        ind_one_rand_lst = ind_mb % tot_message_num
        ind_one = rand_lst[ind_one_rand_lst]
        m[ind_mb, ind_one] = 1
        label[ind_mb] = ind_one
    return m, label

def channel_set_gen_AR(tap_num, h_prev, rho, mul_h_var, if_reset_AR):
    if (type(h_prev) == type(None)) or (if_reset_AR):
        chan_var = mul_h_var * 1 / (2 * tap_num)  # since we are generating real and im. part indep. so 1/2 and we are considering complex, -> 2L generated
        Chan = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2 * tap_num),
                                                                          chan_var * torch.eye(2 * tap_num))
        h_curr = Chan.sample() # this is initial channel
        print('generate actually used meta-training channel which is initial channel: ', h_curr)
    else:
        #print('keep previous channel as AR!')
        chan_var = mul_h_var * 1 / (2 * tap_num)  # since we are generating real and im. part indep. so 1/2 and we are considering complex, -> 2L generated
        Chan = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2 * tap_num),
                                                                          chan_var * torch.eye(2 * tap_num))
        z = Chan.sample()
        h_curr = rho * h_prev + pow((1-pow(rho,2)), 0.5) * z
    return h_curr

def channel_set_gen(num_channels, tap_num): # we can also change mul_h_var here by chaning the code
    channel_list = []
    for ind_channels in range(num_channels):
        chan_var = 1 / (2 * tap_num)  # since we are generating real and im. part indep. so 1/2 and we are considering complex, -> 2L generated
        Chan = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2 * tap_num),
                                                                          chan_var * torch.eye(2 * tap_num))
        h = Chan.sample()
        channel_list.append(h)
    return channel_list



