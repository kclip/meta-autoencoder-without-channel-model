import torch
from utils.basic_funcs import complex_mul_taps
from data_gen.data_set import message_gen
from nets.meta_transmitter import meta_tx
from nets.meta_receiver import meta_rx
from utils.conv_communication_scheme import four_qam_uncoded_modulation, channel_estimation
import numpy as np

def channel(h, x, Noise, device, if_AWGN):
    # x here is transmitted symbol
    # Rayleigh block fading channel
    if if_AWGN:
        pass
    else:
        x = complex_mul_taps(h, x)
    x = x.to(device)
    # noise
    n = torch.zeros(x.shape[0], x.shape[1])
    for noise_batch_ind in range(x.shape[0]):
        n[noise_batch_ind] = Noise.sample()
    n = n.type(torch.FloatTensor).to(device)
    x = x + n  # now this is received signal
    return x

def receiver_loss(out, label):
    rx_loss = torch.nn.functional.cross_entropy(out, label)
    return rx_loss

def transmitter_loss(x_rel, x, reward, relax_sigma):
    # x_rel: constant, x: function (mean), reward: constant that came from receiver
    tx_loss = 0
    for batch_ind in range(x.shape[0]):
        tx_loss += (-0.5) * (1/pow(relax_sigma,2)) * pow(torch.norm(x_rel[batch_ind] - x[batch_ind]), 2) * reward[batch_ind]
    tx_loss = tx_loss/(x.shape[0]) #mean like usual cross entropy does
    return tx_loss

def feedback_from_rx(out, label):
    reward_from_rx = torch.nn.functional.cross_entropy(out, label, reduction='none')
    received_feedback = reward_from_rx.clone().detach()

    return received_feedback


def one_frame_joint_training(args, h, Noise, Noise_relax, curr_tx_net_list, curr_rx_net_list, init_tx_net_list, init_rx_net_list):
    # we only transmit once and update both rx and tx since we are consdiering stochastic encoder always
    # joint training can be obtained via MAML without no inner update
    tx_meta_intermediate = meta_tx(if_relu = args.if_relu)
    rx_meta_intermediate = meta_rx(if_relu = args.if_relu)

    relax_sigma = args.relax_sigma

    m, label = message_gen(args.bit_num, args.pilots_num_meta_train_query)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)

    if args.fix_bpsk_tx:
        actual_transmitted_symbol = four_qam_uncoded_modulation(args.bit_num, args.channel_num,
                                                                label)  # label instead m
    else:
        # tx
        tx_symb_mean, actual_transmitted_symbol = tx_meta_intermediate(m, curr_tx_net_list, args.if_bias, args.device, relax_sigma, Noise_relax)
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)
    # rx
    out = rx_meta_intermediate(received_signal, curr_rx_net_list, args.if_bias, args.device, args.if_RTN)

    # for rx
    loss_rx = receiver_loss(out, label)
    joint_grad_rx = torch.autograd.grad(loss_rx, init_rx_net_list, create_graph=False)

    if args.fix_bpsk_tx:
        joint_grad_tx = None
        loss_tx= 0
    else:
        received_reward_from_rx = feedback_from_rx(out, label)
        loss_tx = transmitter_loss(actual_transmitted_symbol, tx_symb_mean, received_reward_from_rx, args.relax_sigma)
        joint_grad_tx = torch.autograd.grad(loss_tx, init_tx_net_list, create_graph=False, retain_graph=True)

    return joint_grad_rx, joint_grad_tx, float(loss_rx), float(loss_tx)


def one_frame_hybrid_training(args, h, Noise, Noise_relax, para_tx_net_list, para_rx_net_list):
    tx_meta_intermediate = meta_tx(if_relu = args.if_relu)
    rx_meta_intermediate = meta_rx(if_relu = args.if_relu)
    relax_sigma = args.relax_sigma
    m, label = message_gen(args.bit_num, args.pilots_num_meta_train_query) # always send this amount

    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)
    #### we only transmit once and do meta-learning (rx) and joint learning (tx)
    if args.fix_bpsk_tx:
        actual_transmitted_symbol = four_qam_uncoded_modulation(args.bit_num, args.channel_num,
                                                                label)  # label instead m
    else:
        # tx
        tx_symb_mean, actual_transmitted_symbol = tx_meta_intermediate(m, para_tx_net_list, args.if_bias, args.device, relax_sigma, Noise_relax)
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)

    ###### change support num pilots here, transmission is only done once with mb_size_meta_test
    num_pilots_for_meta_train_supp = args.pilots_num_meta_train_supp
    # rx
    for ind_sim_iter_rx in range(args.num_meta_local_updates):
        if ind_sim_iter_rx == 0:
            out = rx_meta_intermediate(received_signal[0:num_pilots_for_meta_train_supp], para_rx_net_list, args.if_bias, args.device, args.if_RTN)
            loss_rx = receiver_loss(out, label[0:num_pilots_for_meta_train_supp])
            local_grad_rx = torch.autograd.grad(loss_rx, para_rx_net_list,
                                                create_graph=True)
            intermediate_updated_para_list_rx = list(
                map(lambda p: p[1] - args.lr_meta_inner * p[0], zip(local_grad_rx, para_rx_net_list)))
            first_loss_curr_rx = float(loss_rx.clone().detach())
        else:
            out = rx_meta_intermediate(received_signal[0:num_pilots_for_meta_train_supp], intermediate_updated_para_list_rx, args.if_bias, args.device, args.if_RTN)
            loss_rx = receiver_loss(out, label[0:num_pilots_for_meta_train_supp])
            local_grad_rx = torch.autograd.grad(loss_rx, intermediate_updated_para_list_rx,
                                                create_graph=True)
            intermediate_updated_para_list_rx = list(
                map(lambda p: p[1] - args.lr_meta_inner * p[0], zip(local_grad_rx, intermediate_updated_para_list_rx)))

    ### now meta-gradient
    if args.separate_meta_training_support_query_set:
        end_ind_for_supp = num_pilots_for_meta_train_supp
    else: # use whole for query
        end_ind_for_supp = 0

    out = rx_meta_intermediate(received_signal[end_ind_for_supp:], intermediate_updated_para_list_rx, args.if_bias, args.device, args.if_RTN)
    loss_rx_after_local_adaptation = receiver_loss(out, label[end_ind_for_supp:])
    meta_grad_rx = torch.autograd.grad(loss_rx_after_local_adaptation, para_rx_net_list, create_graph=False)

    # use all transmission blocks in one frame for joint training of encoder
    end_ind_for_supp = 0
    if args.separate_meta_training_support_query_set: # we need to get out for whole messages
        out = rx_meta_intermediate(received_signal, intermediate_updated_para_list_rx,
                                   args.if_bias, args.device, args.if_RTN)
    else:
        pass

    if args.fix_bpsk_tx:
        loss_tx = 0
        joint_grad_tx = None
    else:
        # joint grad. for tx
        received_reward_from_rx = feedback_from_rx(out, label[end_ind_for_supp:]) # feedback with adapted rx
        loss_tx = transmitter_loss(actual_transmitted_symbol[end_ind_for_supp:], tx_symb_mean[end_ind_for_supp:], received_reward_from_rx, args.relax_sigma)
        joint_grad_tx = torch.autograd.grad(loss_tx, para_tx_net_list, create_graph=False, retain_graph=True)

    return meta_grad_rx, joint_grad_tx, first_loss_curr_rx, float(loss_tx), float(loss_rx_after_local_adaptation), float(loss_tx)


def one_iter_mmse_ch_est(args, h, Noise, num_pilots_test):
    m, label = message_gen(args.bit_num, num_pilots_test)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)
    actual_transmitted_symbol = four_qam_uncoded_modulation(args.bit_num, args.channel_num, label) # label instead m
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)
    est_h = channel_estimation(args, actual_transmitted_symbol, received_signal, args.tap_num)
    h_numpy = h.cpu().numpy()
    h_complex = np.zeros((args.tap_num, 1), dtype=complex)
    for ind_h in range(args.tap_num):
        h_complex[ind_h] = h_numpy[2 * ind_h] + h_numpy[2 * ind_h + 1] * 1j
    error_h = np.matmul(np.conj(np.transpose(h_complex-est_h)), h_complex-est_h)
    return est_h, error_h

def one_frame_conventional_training_tx_nn_rx_nn(args, h, Noise, Noise_relax, tx_net_for_testtraining, rx_net_for_testtraining, rx_testtraining_optimiser, epochs, num_pilots_test_in_one_mb):
    # only for runtime
    relax_sigma = args.relax_sigma
    m, label = message_gen(args.bit_num, num_pilots_test_in_one_mb)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)
    tx_net_for_testtraining.zero_grad()
    # tx
    tx_symb_mean, actual_transmitted_symbol = tx_net_for_testtraining(m, args.device, relax_sigma, Noise_relax)
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)

    #### we can do multiple training with given received_signal
    for rx_training_iter in range(args.fix_tx_multi_adapt_rx_iter_num):
        rx_net_for_testtraining.zero_grad()
        received_signal_curr_mb = received_signal
        label_curr_mb = label
        out = rx_net_for_testtraining(received_signal_curr_mb, args.if_RTN, args.device)
        loss_rx = receiver_loss(out, label_curr_mb)

        loss_rx.backward()

        if args.if_test_training_adam:
            if args.if_adam_after_sgd:
                if epochs < args.num_meta_local_updates:
                    for f in rx_net_for_testtraining.parameters():
                        if f.grad is not None:
                            f.data.sub_(f.grad.data * args.lr_meta_inner)
                elif epochs == args.num_meta_local_updates:
                    rx_testtraining_optimiser = torch.optim.Adam(rx_net_for_testtraining.parameters(),
                                                              args.lr_testtraining)
                    rx_testtraining_optimiser.step()
                else:
                    rx_testtraining_optimiser.step()
            else:
                rx_testtraining_optimiser.step()
        else:
            for f in rx_net_for_testtraining.parameters():
                if f.grad is not None:
                    f.data.sub_(f.grad.detach() * args.lr_testtraining)
    loss_tx = 0

    return rx_testtraining_optimiser, float(loss_rx), float(loss_tx)

def one_frame_conventional_training_tx_bpsk_rx_nn(args, h, Noise, rx_net_for_testtraining, rx_testtraining_optimiser, num_pilots_test_in_one_mb):
    # only for runtime
    m, label = message_gen(args.bit_num, num_pilots_test_in_one_mb)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)

    tx_net_for_testtraining = None # we do not need tx neural net
    # tx
    actual_transmitted_symbol = four_qam_uncoded_modulation(args.bit_num, args.channel_num, label)  # label instead m
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)

    #### we can do multiple training with given received_signal
    for rx_training_iter in range(args.fix_tx_multi_adapt_rx_iter_num):
        rx_net_for_testtraining.zero_grad()

        received_signal_curr_mb = received_signal
        label_curr_mb = label

        out = rx_net_for_testtraining(received_signal_curr_mb, args.if_RTN, args.device)
        loss_rx = receiver_loss(out, label_curr_mb)

        loss_rx.backward()

        if args.if_test_training_adam:
            if args.if_adam_after_sgd:
                if rx_training_iter < args.num_meta_local_updates:
                    for f in rx_net_for_testtraining.parameters():
                        if f.grad is not None:
                            f.data.sub_(f.grad.data * args.lr_meta_inner)
                elif rx_training_iter == args.num_meta_local_updates:
                    rx_testtraining_optimiser = torch.optim.Adam(rx_net_for_testtraining.parameters(),
                                                              args.lr_testtraining)
                    rx_testtraining_optimiser.step()
                else:
                    rx_testtraining_optimiser.step()
            else:
                rx_testtraining_optimiser.step()
        else:
            for f in rx_net_for_testtraining.parameters():
                if f.grad is not None:
                    f.data.sub_(f.grad.detach() * args.lr_testtraining)
    loss_tx = 0
    return rx_testtraining_optimiser, float(loss_rx), float(loss_tx)