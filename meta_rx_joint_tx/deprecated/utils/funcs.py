import torch
from data_gen.data_set import message_gen
from nets.meta_transmitter import meta_tx
from nets.meta_receiver import meta_rx
from utils.basic_funcs import complex_mul_taps

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
    tx_loss = tx_loss/(x.shape[0]) #mean like normal cross entropy does
    return tx_loss

def feedback_from_rx(out, label, Noise_feedback, device):
    reward_from_rx = torch.nn.functional.cross_entropy(out, label, reduction='none')
    w_feedback = torch.zeros(reward_from_rx.shape[0])
    if Noise_feedback is not None:
        for batch_ind_reward in range(reward_from_rx.shape[0]):
            w_feedback[batch_ind_reward] = Noise_feedback.sample()
    else:
        pass
    w_feedback = w_feedback.type(torch.FloatTensor).to(device)
    received_feedback = (reward_from_rx + w_feedback).clone().detach()
    return received_feedback


def one_iter_rx_meta_tx_joint_sim_fix_stoch_encoder(args, h, Noise, Noise_relax, Noise_feedback, para_tx_net_list, para_rx_net_list):
    tx_meta_intermediate = meta_tx(if_relu = args.if_relu)
    rx_meta_intermediate = meta_rx(if_relu = args.if_relu)
    relax_sigma = args.relax_sigma
    m, label = message_gen(args.bit_num, args.mb_size)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)
    #### we only transmit once and do meta-learning (rx) and joint learning (tx)
    # tx
    tx_symb_mean, actual_transmitted_symbol = tx_meta_intermediate(m, para_tx_net_list, args.if_bias, args.device, relax_sigma, Noise_relax)
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)
    # rx
    for ind_sim_iter_rx in range(args.num_meta_local_updates):
        if ind_sim_iter_rx == 0:
            out = rx_meta_intermediate(received_signal, para_rx_net_list, args.if_bias, args.device, args.if_RTN)
            loss_rx = receiver_loss(out, label)
            local_grad_rx = torch.autograd.grad(loss_rx, para_rx_net_list,
                                                create_graph=True)
            intermediate_updated_para_list_rx = list(
                map(lambda p: p[1] - args.lr_meta_inner * p[0], zip(local_grad_rx, para_rx_net_list)))
            first_loss_curr_rx = float(loss_rx.clone().detach())
        else:
            out = rx_meta_intermediate(received_signal, intermediate_updated_para_list_rx, args.if_bias, args.device, args.if_RTN)
            loss_rx = receiver_loss(out, label)
            local_grad_rx = torch.autograd.grad(loss_rx, intermediate_updated_para_list_rx,
                                                create_graph=True)
            intermediate_updated_para_list_rx = list(
                map(lambda p: p[1] - args.lr_meta_inner * p[0], zip(local_grad_rx, intermediate_updated_para_list_rx)))

    ### now meta-gradient
    out = rx_meta_intermediate(received_signal, intermediate_updated_para_list_rx, args.if_bias, args.device, args.if_RTN)
    loss_rx_after_local_adaptation = receiver_loss(out, label)
    meta_grad_rx = torch.autograd.grad(loss_rx_after_local_adaptation, para_rx_net_list, create_graph=False)

    # joint grad. for tx
    received_reward_from_rx = feedback_from_rx(out, label, Noise_feedback, args.device) # feedback with adapted rx
    loss_tx = transmitter_loss(actual_transmitted_symbol, tx_symb_mean, received_reward_from_rx, args.relax_sigma)
    joint_grad_tx = torch.autograd.grad(loss_tx, para_tx_net_list, create_graph=False, retain_graph=True)

    return meta_grad_rx, joint_grad_tx, first_loss_curr_rx, float(loss_tx), float(loss_rx_after_local_adaptation), float(loss_tx)


def one_iter_joint_training_sim_fix_stoch_encoder(args, h, Noise, Noise_relax, Noise_feedback, curr_tx_net_list, curr_rx_net_list, init_tx_net_list, init_rx_net_list, if_local_update, remove_dependency_to_updated_para_tmp, inner_loop):
    # given net list, run meta_net
    # we only transmit once and update both rx and tx since we are consdiering stochastic encoder always
    tx_meta_intermediate = meta_tx(if_relu = args.if_relu)
    rx_meta_intermediate = meta_rx(if_relu = args.if_relu)
    relax_sigma = args.relax_sigma

    m, label = message_gen(args.bit_num, args.mb_size)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)
    # tx
    tx_symb_mean, actual_transmitted_symbol = tx_meta_intermediate(m, curr_tx_net_list, args.if_bias, args.device, relax_sigma, Noise_relax)
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)
    # rx
    out = rx_meta_intermediate(received_signal, curr_rx_net_list, args.if_bias, args.device, args.if_RTN)
    # for rx
    loss_rx = receiver_loss(out, label)
    joint_grad_rx = torch.autograd.grad(loss_rx, init_rx_net_list, create_graph=False)
    received_reward_from_rx = feedback_from_rx(out, label, Noise_feedback, args.device)
    loss_tx = transmitter_loss(actual_transmitted_symbol, tx_symb_mean, received_reward_from_rx, args.relax_sigma)
    joint_grad_tx = torch.autograd.grad(loss_tx, init_tx_net_list, create_graph=False, retain_graph=True)

    return joint_grad_rx, joint_grad_tx, float(loss_rx), float(loss_tx)




def one_iter_sim_fix_stoch_encoder(args, h, Noise, Noise_relax, Noise_feedback, tx_net_for_testtraining, rx_net_for_testtraining, tx_testtraining_optimiser, rx_testtraining_optimiser, epochs):
    relax_sigma = args.relax_sigma
    m, label = message_gen(args.bit_num, args.mb_size)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)

    tx_net_for_testtraining.zero_grad()
    rx_net_for_testtraining.zero_grad()

    # tx
    tx_symb_mean, actual_transmitted_symbol = tx_net_for_testtraining(m, args.device, relax_sigma, Noise_relax)
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)
    # rx
    out = rx_net_for_testtraining(received_signal, args.if_RTN, args.device)

    loss_rx = receiver_loss(out, label)
    loss_rx.backward()
    received_reward_from_rx = feedback_from_rx(out, label, Noise_feedback, args.device)
    loss_tx = transmitter_loss(actual_transmitted_symbol, tx_symb_mean, received_reward_from_rx,
                               args.relax_sigma)
    loss_tx.backward()

    if args.if_test_training_adam:
        if args.if_adam_after_sgd:
            if epochs < args.num_meta_local_updates:
                for f in rx_net_for_testtraining.parameters():
                    if f.grad is not None:
                        f.data.sub_(f.grad.data * args.lr_meta_inner)
                for f in tx_net_for_testtraining.parameters():
                    if f.grad is not None:
                        f.data.sub_(f.grad.data * args.lr_meta_inner)
            elif epochs == args.num_meta_local_updates:
                rx_testtraining_optimiser = torch.optim.Adam(rx_net_for_testtraining.parameters(),
                                                          args.lr_testtraining)
                rx_testtraining_optimiser.step()
                tx_testtraining_optimiser = torch.optim.Adam(tx_net_for_testtraining.parameters(),
                                                             args.lr_testtraining)
                tx_testtraining_optimiser.step()
            else:
                rx_testtraining_optimiser.step()
                tx_testtraining_optimiser.step()
        else:
            rx_testtraining_optimiser.step()
            tx_testtraining_optimiser.step()
    else:
        for f in rx_net_for_testtraining.parameters():
            if f.grad is not None:
                f.data.sub_(f.grad.detach() * args.lr_testtraining)
        for f in tx_net_for_testtraining.parameters():
            if f.grad is not None:
                f.data.sub_(f.grad.detach() * args.lr_testtraining)

    return rx_testtraining_optimiser, tx_testtraining_optimiser, float(loss_rx), float(loss_tx)


def one_iter_sim_fix_stoch_encoder_from_rx_meta_tx_joint(args, h, Noise, Noise_relax, Noise_feedback, tx_net_for_testtraining, rx_net_for_testtraining, tx_testtraining_optimiser, rx_testtraining_optimiser, epochs):
    relax_sigma = args.relax_sigma
    m, label = message_gen(args.bit_num, args.mb_size)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)

    tx_net_for_testtraining.zero_grad()
    rx_net_for_testtraining.zero_grad()
    # tx
    tx_symb_mean, actual_transmitted_symbol = tx_net_for_testtraining(m, args.device, relax_sigma, Noise_relax)
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise, args.device, args.if_AWGN)
    # rx
    out = rx_net_for_testtraining(received_signal, args.if_RTN, args.device)

    loss_rx = receiver_loss(out, label)
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

    if args.fix_joint_trained_tx_only_adapt_meta_trained_rx:
        tx_testtraining_optimiser = None
        loss_tx = 0
    else:
        out = rx_net_for_testtraining(received_signal, args.if_RTN, args.device)
        received_reward_from_rx = feedback_from_rx(out, label, Noise_feedback, args.device)
        loss_tx = transmitter_loss(actual_transmitted_symbol, tx_symb_mean, received_reward_from_rx,
                                   args.relax_sigma)
        loss_tx.backward()
        if args.if_test_training_adam:
            if args.if_adam_after_sgd:
                if epochs < args.num_meta_local_updates:
                    for f in tx_net_for_testtraining.parameters():
                        if f.grad is not None:
                            f.data.sub_(f.grad.data * args.lr_meta_inner)
                elif epochs == args.num_meta_local_updates:
                    tx_testtraining_optimiser = torch.optim.Adam(tx_net_for_testtraining.parameters(),
                                                                 args.lr_testtraining)
                    tx_testtraining_optimiser.step()
                else:
                    tx_testtraining_optimiser.step()
            else:
                tx_testtraining_optimiser.step()
        else:
            for f in tx_net_for_testtraining.parameters():
                if f.grad is not None:
                    f.data.sub_(f.grad.detach() * args.lr_testtraining)


    return rx_testtraining_optimiser, tx_testtraining_optimiser, float(loss_rx), float(loss_tx)



