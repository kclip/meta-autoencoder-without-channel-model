import torch
from utils.funcs import one_iter_joint_training_sim_fix_stoch_encoder, one_iter_rx_meta_tx_joint_sim_fix_stoch_encoder
from data_gen.data_set import channel_set_gen_AR
from utils.basic_funcs import reset_randomness

def multi_task_learning(args, tx_net,rx_net, h_list_meta, writer_meta_training, Noise, Noise_relax, Noise_feedback, PATH_before_adapt_rx_intermediate, PATH_before_adapt_tx_intermediate):
    meta_optimiser_tx = torch.optim.Adam(tx_net.parameters(), args.lr_meta_update)
    meta_optimiser_rx = torch.optim.Adam(rx_net.parameters(), args.lr_meta_update)
    h_list_train = h_list_meta[:args.num_channels_meta]

    if args.if_fix_random_seed:
        random_seed_init = args.random_seed + 99999
    else:
        pass
    previous_channel = None

    for epochs in range(args.num_epochs_meta_train):
        if epochs % 100 == 0:
            curr_path_rx = PATH_before_adapt_rx_intermediate + str(epochs)
            curr_path_tx = PATH_before_adapt_tx_intermediate + str(epochs)
            torch.save(rx_net.state_dict(), curr_path_rx)
            torch.save(tx_net.state_dict(), curr_path_tx)
            print('stochactic meta-learning epoch', epochs)
        first_loss_rx = 0
        second_loss_rx = 0
        first_loss_tx = 0
        second_loss_tx = 0
        iter_in_sampled_device = 0  # for averaging meta-devices
        for ind_meta_dev in range(args.tasks_per_metaupdate):
            if args.if_always_generate_new_meta_training_channels:
                if args.if_fix_random_seed:
                    reset_randomness(random_seed_init) # to make noise same as much as possible for joint and meta
                    random_seed_init += 1
                else:
                    pass
                if args.if_Rayleigh_channel_model_AR:
                    if epochs % args.keep_AR_period == 0:
                        h_var_dist = torch.distributions.uniform.Uniform(torch.FloatTensor([args.mul_h_var_min]), torch.FloatTensor([args.mul_h_var_max]))
                        mul_h_var = h_var_dist.sample()
                        if_reset_AR = True
                    else:
                        if_reset_AR = False
                    current_channel = channel_set_gen_AR(args.tap_num, previous_channel, args.rho, mul_h_var, if_reset_AR)  # num_channels = 1 since we are generating per channel
                    previous_channel = current_channel
                else:
                    raise NotImplementedError
            else:
                if args.if_fix_random_seed:
                    reset_randomness(random_seed_init) # to make noise same as much as possible for joint and meta
                    random_seed_init += 1
                else:
                    pass
                # during this, meta-gradients are accumulated
                channel_list_total = torch.randperm(len(h_list_train))  # sampling with replacement
                current_channel_ind = channel_list_total[
                    ind_meta_dev]  # randomly sample meta-batches (no rep. inside meta-batch)
                current_channel = h_list_train[current_channel_ind]

            if args.if_joint_training:
                iter_in_sampled_device, first_loss_curr_rx, first_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx = joint_training(args, iter_in_sampled_device,
                                                                                 tx_net, rx_net,
                                                                                 current_channel, Noise, Noise_relax, Noise_feedback)
            elif args.if_joint_training_tx_meta_training_rx:
                iter_in_sampled_device, first_loss_curr_rx, first_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx = maml_for_rx_joint_for_tx(
                    args, iter_in_sampled_device,
                    tx_net, rx_net,
                    current_channel, Noise, Noise_relax, Noise_feedback)
            else:  # maml
                raise NotImplementedError
            first_loss_rx = first_loss_rx + first_loss_curr_rx
            second_loss_rx = second_loss_rx + second_loss_curr_rx
            first_loss_tx = first_loss_tx + first_loss_curr_tx
            second_loss_tx = second_loss_tx + second_loss_curr_tx
        first_loss_tx = first_loss_tx / args.tasks_per_metaupdate
        second_loss_tx = second_loss_tx / args.tasks_per_metaupdate
        first_loss_rx = first_loss_rx / args.tasks_per_metaupdate
        second_loss_rx = second_loss_rx / args.tasks_per_metaupdate
        writer_meta_training.add_scalar('first rx loss', first_loss_rx, epochs)
        writer_meta_training.add_scalar('first tx loss', first_loss_tx, epochs)
        writer_meta_training.add_scalar('second rx loss', second_loss_rx, epochs)
        writer_meta_training.add_scalar('second tx loss', second_loss_tx, epochs)
        meta_optimiser_rx.zero_grad()
        meta_optimiser_tx.zero_grad()
        for f in rx_net.parameters():
            f.grad = f.total_grad.clone() / args.tasks_per_metaupdate
        for f in tx_net.parameters():
            f.grad = f.total_grad.clone() / args.tasks_per_metaupdate
        meta_optimiser_rx.step()  # Adam
        meta_optimiser_tx.step()  # Adam

def joint_training(args, iter_in_sampled_device, tx_net, rx_net, current_channel, Noise, Noise_relax, Noise_feedback):
    tx_net.zero_grad()
    rx_net.zero_grad()
    para_list_from_tx_net = list(map(lambda p: p[0], zip(tx_net.parameters())))
    para_list_from_rx_net = list(map(lambda p: p[0], zip(rx_net.parameters())))
    #### get gradients for joint training
    if_local_update = False
    remove_dependency_to_updated_para = False
    inner_loop = None # no need here, it is only for computing exact score function of tx meta-gradient

    if args.tx_rx_training_mode == 0:
        joint_grad_rx, joint_grad_tx, second_loss_curr_rx, second_loss_curr_tx = one_iter_joint_training_sim_fix_stoch_encoder(args, current_channel, Noise, Noise_relax, Noise_feedback,
                                                                              para_list_from_tx_net,
                                                                              para_list_from_rx_net,
                                                                              para_list_from_tx_net,
                                                                              para_list_from_rx_net,
                                                                              if_local_update,
                                                                              remove_dependency_to_updated_para,
                                                                              inner_loop)
    else:
        raise NotImplementedError

    ##### accumulate gradients for joint training over different channels
    ind_f_para_list = 0
    for f in rx_net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = joint_grad_rx[ind_f_para_list].clone().detach()  # f.grad
        else:
            f.total_grad = f.total_grad + joint_grad_rx[ind_f_para_list].clone().detach()  # f.grad
        ind_f_para_list += 1

    ind_f_para_list = 0
    for f in tx_net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = joint_grad_tx[ind_f_para_list].clone().detach()  # f.grad
        else:
            f.total_grad = f.total_grad + joint_grad_tx[ind_f_para_list].clone().detach()  # f.grad
        ind_f_para_list += 1
    iter_in_sampled_device = iter_in_sampled_device + 1

    return iter_in_sampled_device, second_loss_curr_rx, second_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx


def maml_for_rx_joint_for_tx(args, iter_in_sampled_device, tx_net, rx_net, current_channel, Noise, Noise_relax, Noise_feedback):
    tx_net.zero_grad()
    rx_net.zero_grad()
    para_list_from_tx_net = list(map(lambda p: p[0], zip(tx_net.parameters())))
    para_list_from_rx_net = list(map(lambda p: p[0], zip(rx_net.parameters())))

    ####here comes from ftn.
    if args.tx_rx_training_mode == 0:
        meta_grad_rx, joint_grad_tx, first_loss_curr_rx, first_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx = one_iter_rx_meta_tx_joint_sim_fix_stoch_encoder(args, current_channel, Noise, Noise_relax, Noise_feedback,
                                                        para_list_from_tx_net, para_list_from_rx_net)


    else:
        print('we have only 3 modes, 0, 1, 2 currently')
        raise NotImplementedError
    ##### accumulate meta-gradients over different channels (meta-training channels)
    ind_f_para_list = 0
    for f in rx_net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = meta_grad_rx[ind_f_para_list].clone().detach()  # f.grad
        else:
            f.total_grad = f.total_grad + meta_grad_rx[ind_f_para_list].clone().detach()  # f.grad
        ind_f_para_list += 1
    ind_f_para_list = 0
    for f in tx_net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = joint_grad_tx[ind_f_para_list].clone().detach()  # f.grad
        else:
            f.total_grad = f.total_grad + joint_grad_tx[ind_f_para_list].clone().detach()  # f.grad
        ind_f_para_list += 1
    iter_in_sampled_device = iter_in_sampled_device + 1

    return iter_in_sampled_device, first_loss_curr_rx, first_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx


