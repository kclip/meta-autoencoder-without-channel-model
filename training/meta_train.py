import torch
from utils.funcs import one_frame_joint_training, one_frame_hybrid_training
from data_gen.data_set import channel_set_gen_AR
from utils.basic_funcs import reset_randomness
from utils.test_bler_with_adaptation import test_with_adapt_compact_during_online_meta_training, test_conven_commun_during_online_meta_training
import os
import scipy.io as sio

def multi_task_learning(args, common_dir, tx_net,rx_net, writer_meta_training, Noise, Noise_relax, actual_channel_num, PATH_before_adapt_rx_intermediate, PATH_before_adapt_tx_intermediate, rx_net_for_testtraining, tx_net_for_testtraining):
    meta_optimiser_tx = torch.optim.Adam(tx_net.parameters(), args.lr_meta_update)
    meta_optimiser_rx = torch.optim.Adam(rx_net.parameters(), args.lr_meta_update)

    test_result_PATH_per_meta_training_test_bler_dir = common_dir + 'test_result_during_meta_training/'
    save_test_result_dict_total_per_meta_training_test_bler = {}

    if args.if_fix_random_seed:
        random_seed_init = args.random_seed + 99999
    else:
        pass
    # for online
    previous_channel = []
    for ind_dev in range(args.tasks_per_metaupdate):
        previous_channel.append(None)

    test_bler_per_meta_training_epochs = []
    channel_per_meta_training_epochs = []

    second_loss_rx_best_for_stopping_criteria = 99999999999

    for epochs in range(args.num_epochs_meta_train):
        if epochs % args.meta_tr_epoch_num_for_test == 0:
            curr_path_rx = PATH_before_adapt_rx_intermediate + str(epochs)
            curr_path_tx = PATH_before_adapt_tx_intermediate + str(epochs)
            torch.save(rx_net.state_dict(), curr_path_rx)
            torch.save(tx_net.state_dict(), curr_path_tx)
            print('stochactic meta-learning epoch', epochs)
            if args.see_test_bler_during_meta_update:
                # see test bler here
                test_snr_range = [args.Eb_over_N_db_test]
                num_pilots_test = args.pilots_num_meta_test
                PATH_before_adapt_tx = curr_path_tx
                PATH_before_adapt_rx = curr_path_rx

                if args.if_get_conven_commun_performance:
                    test_bler_mean_curr_epoch_conven_approach, ch_est_error_avg = test_conven_commun_during_online_meta_training(args,
                                                                                                    test_snr_range,
                                                                                                    num_pilots_test,
                                                                                                    Noise,
                                                                                                    actual_channel_num)
                    writer_meta_training.add_scalar('conv. approach test bler during meta-training',
                                                    test_bler_mean_curr_epoch_conven_approach, epochs)
                    print('conven bler: ', test_bler_mean_curr_epoch_conven_approach)
                    print('conven ch. est: ', ch_est_error_avg)
                    print('as this is for BPSK with maximum likelihood decoder, we only need this once so we stop the code here')
                    dfdfdfdfd
                else:
                    pass

                test_bler_mean_curr_epoch = test_with_adapt_compact_during_online_meta_training(args, common_dir,
                                                                                                epochs,
                                                                                                test_snr_range,
                                                                                                num_pilots_test,
                                                                                                tx_net_for_testtraining,
                                                                                                rx_net_for_testtraining,
                                                                                                Noise,
                                                                                                Noise_relax,
                                                                                                actual_channel_num,
                                                                                                PATH_before_adapt_tx,
                                                                                                PATH_before_adapt_rx)
                writer_meta_training.add_scalar('test bler during meta-training',
                                       test_bler_mean_curr_epoch, epochs)
                test_bler_per_meta_training_epochs.append(test_bler_mean_curr_epoch)

        first_loss_rx = 0
        second_loss_rx = 0
        first_loss_tx = 0
        second_loss_tx = 0
        iter_in_sampled_device = 0  # for averaging meta-devices
        for ind_meta_dev in range(args.tasks_per_metaupdate):
            if args.if_fix_random_seed:
                reset_randomness(random_seed_init) # to make noise same as much as possible for joint and meta
                random_seed_init += 1
            else:
                pass
            if args.if_Rayleigh_channel_model_AR:
                current_channel = channel_set_gen_AR(args.tap_num, previous_channel[ind_meta_dev], args.rho)  # num_channels = 1 since we are generating per channel
                previous_channel[ind_meta_dev] = current_channel
            else:
                raise NotImplementedError


            if args.if_joint_training:
                iter_in_sampled_device, first_loss_curr_rx, first_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx = joint_training(args, iter_in_sampled_device,
                                                                                 tx_net, rx_net,
                                                                                 current_channel, Noise, Noise_relax)
            elif args.if_joint_training_tx_meta_training_rx:
                iter_in_sampled_device, first_loss_curr_rx, first_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx = maml_for_rx_joint_for_tx(
                    args, iter_in_sampled_device,
                    tx_net, rx_net,
                    current_channel, Noise, Noise_relax)
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
        if args.if_TB_loss_ignore:
            pass
        else:
            writer_meta_training.add_scalar('first rx loss', first_loss_rx, epochs)
            writer_meta_training.add_scalar('first tx loss', first_loss_tx, epochs)
            writer_meta_training.add_scalar('second rx loss', second_loss_rx, epochs)
            writer_meta_training.add_scalar('second tx loss', second_loss_tx, epochs)

        if args.if_use_stopping_criteria_during_meta_training:
            if second_loss_rx < second_loss_rx_best_for_stopping_criteria:
                curr_path_rx_best_training_loss = PATH_before_adapt_rx_intermediate + 'best_model_based_on_meta_training_loss'
                curr_path_tx_best_training_loss = PATH_before_adapt_tx_intermediate + 'best_model_based_on_meta_training_loss'
                torch.save(rx_net.state_dict(), curr_path_rx_best_training_loss)
                torch.save(tx_net.state_dict(), curr_path_tx_best_training_loss)
            else:
                pass
        else:
            pass


        meta_optimiser_rx.zero_grad()
        meta_optimiser_tx.zero_grad()
        # rx meta-update
        for f in rx_net.parameters():
            f.grad = f.total_grad.clone() / args.tasks_per_metaupdate
        meta_optimiser_rx.step()  # Adam
        # tx meta-update
        if args.fix_bpsk_tx: # nothing to meta-learn for tx since tx is BPSK encoder
            pass
        else:
            for f in tx_net.parameters():
                f.grad = f.total_grad.clone() / args.tasks_per_metaupdate
            meta_optimiser_tx.step()  # Adam

        if epochs % args.meta_tr_epoch_num_for_test == 0:
            os.makedirs(test_result_PATH_per_meta_training_test_bler_dir + 'epochs/' + str(epochs) + '/')
            test_result_PATH_per_meta_training_test_bler = test_result_PATH_per_meta_training_test_bler_dir + 'epochs/' + str(epochs) + '/' + 'test_result_per_meta_training_epochs.mat'
            save_test_result_dict_total_per_meta_training_test_bler[
                'test_bler_during_meta_training'] = test_bler_per_meta_training_epochs
            sio.savemat(test_result_PATH_per_meta_training_test_bler,
                        save_test_result_dict_total_per_meta_training_test_bler)
        else:
            pass



    return test_bler_per_meta_training_epochs, channel_per_meta_training_epochs


def joint_training(args, iter_in_sampled_device, tx_net, rx_net, current_channel, Noise, Noise_relax):
    tx_net.zero_grad()
    rx_net.zero_grad()
    para_list_from_tx_net = list(map(lambda p: p[0], zip(tx_net.parameters())))
    para_list_from_rx_net = list(map(lambda p: p[0], zip(rx_net.parameters())))
    #### get gradients for joint training
    joint_grad_rx, joint_grad_tx, second_loss_curr_rx, second_loss_curr_tx = one_frame_joint_training(args, current_channel, Noise, Noise_relax,
                                                                          para_list_from_tx_net,
                                                                          para_list_from_rx_net,
                                                                          para_list_from_tx_net,
                                                                          para_list_from_rx_net)

    ##### accumulate gradients for joint training over different channels
    # rx
    ind_f_para_list = 0
    for f in rx_net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = joint_grad_rx[ind_f_para_list].clone().detach()
        else:
            f.total_grad = f.total_grad + joint_grad_rx[ind_f_para_list].clone().detach()
        ind_f_para_list += 1

    # tx
    if args.fix_bpsk_tx:
        pass
    else:
        ind_f_para_list = 0
        for f in tx_net.parameters():
            if iter_in_sampled_device == 0:
                f.total_grad = joint_grad_tx[ind_f_para_list].clone().detach()
            else:
                f.total_grad = f.total_grad + joint_grad_tx[ind_f_para_list].clone().detach()
            ind_f_para_list += 1
    iter_in_sampled_device = iter_in_sampled_device + 1

    return iter_in_sampled_device, second_loss_curr_rx, second_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx

def maml_for_rx_joint_for_tx(args, iter_in_sampled_device, tx_net, rx_net, current_channel, Noise, Noise_relax):
    tx_net.zero_grad()
    rx_net.zero_grad()
    para_list_from_tx_net = list(map(lambda p: p[0], zip(tx_net.parameters())))
    para_list_from_rx_net = list(map(lambda p: p[0], zip(rx_net.parameters())))

    ####here comes from ftn.
    meta_grad_rx, joint_grad_tx, first_loss_curr_rx, first_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx = one_frame_hybrid_training(args, current_channel, Noise, Noise_relax, para_list_from_tx_net, para_list_from_rx_net)
    ##### accumulate meta-gradients over different channels (meta-training channels)
    ind_f_para_list = 0
    for f in rx_net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = meta_grad_rx[ind_f_para_list].clone().detach()  # f.grad
        else:
            f.total_grad = f.total_grad + meta_grad_rx[ind_f_para_list].clone().detach()  # f.grad
        ind_f_para_list += 1
    ind_f_para_list = 0

    if args.fix_bpsk_tx:
        pass
    else:
        for f in tx_net.parameters():
            if iter_in_sampled_device == 0:
                f.total_grad = joint_grad_tx[ind_f_para_list].clone().detach()  # f.grad
            else:
                f.total_grad = f.total_grad + joint_grad_tx[ind_f_para_list].clone().detach()  # f.grad
            ind_f_para_list += 1
    iter_in_sampled_device = iter_in_sampled_device + 1

    return iter_in_sampled_device, first_loss_curr_rx, first_loss_curr_tx, second_loss_curr_rx, second_loss_curr_tx




