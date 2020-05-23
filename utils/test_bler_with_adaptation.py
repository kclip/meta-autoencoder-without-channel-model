import torch
from utils.basic_funcs import reset_randomness
from data_gen.data_set import channel_set_gen
from training.train import test_training, test_training_conven_commun
from training.test import test_per_channel_per_snr, test_per_channel_per_snr_conven_approach
import pickle
from torch.utils.tensorboard import SummaryWriter
import os
import scipy.io as sio


def test_with_adapt(args, common_dir, common_dir_over_multi_rand_seeds, test_snr_range, test_num_pilots_available, meta_training_epoch_for_test, tx_net_for_testtraining, rx_net_for_testtraining, Noise, Noise_relax, actual_channel_num, save_test_result_dict_total, test_result_all_PATH_for_all_meta_training_epochs, PATH_before_adapt_tx, PATH_before_adapt_rx):
    # generate or load test channels
    if args.path_for_test_channels is None:
        if args.if_fix_random_seed:
            reset_randomness(args.random_seed + 11)
        print('generate test channels')
        h_list_test = channel_set_gen(args.num_channels_test, args.tap_num, args.if_toy)
        h_list_test_path = common_dir + 'test_channels/' + 'test_channels.pckl'
        f_test_channels = open(h_list_test_path, 'wb')
        pickle.dump(h_list_test, f_test_channels)
        f_test_channels.close()
    else:
        print('load previously generated channels')
        h_list_test_path = args.path_for_test_channels + '/' + 'test_channels.pckl'
        f_test_channels = open(h_list_test_path, 'rb')
        h_list_test = pickle.load(f_test_channels)
        f_test_channels.close()

    if len(h_list_test) > args.num_channels_test:
        h_list_test = h_list_test[:args.num_channels_test]
    print('used test channels', h_list_test)

    dir_test = common_dir + 'TB/' + 'test'
    writer_test = SummaryWriter(dir_test)

    total_total_block_error_rate = torch.zeros(args.num_channels_test, len(test_snr_range), len(test_num_pilots_available), len(meta_training_epoch_for_test))
    ind_meta_training_epoch = 0
    for meta_training_epochs in meta_training_epoch_for_test:
        if common_dir_over_multi_rand_seeds is not None:
            os.makedirs(common_dir_over_multi_rand_seeds  + 'test_result_after_meta_training/' + 'iter/' + str(args.fix_tx_multi_adapt_rx_iter_num ) +'rho/' + str(args.rho) + '/rand_seeds/' + str(args.random_seed) + '/test_result/' + 'with_meta_training_epoch/' + str(
                meta_training_epochs) + '/')
            test_result_all_PATH_per_rand_seeds = common_dir_over_multi_rand_seeds + 'test_result_after_meta_training/'+ 'iter/' + str(args.fix_tx_multi_adapt_rx_iter_num ) + 'rho/' + str(args.rho) + '/rand_seeds/' + str(args.random_seed) + '/test_result/' + 'with_meta_training_epoch/' + str(
                meta_training_epochs) + '/' + 'test_result.mat'

        save_test_result_dict = {}
        # start with given initialization
        print('start adaptation with test set')
        total_block_error_rate = torch.zeros(args.num_channels_test, len(test_snr_range), len(test_num_pilots_available))
        ind_num_pilots_test = 0
        for num_pilots_test in test_num_pilots_available:
            # reset again # to make fair comp. per adapt. and per meta-training epochs
            if args.if_fix_random_seed:
                reset_randomness(args.random_seed + 2)
            print('curr pilots num: ', num_pilots_test)
            os.makedirs(common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + 'tx/' + 'after_adapt/' + str(num_pilots_test) + '_num_pilots_test/')
            os.makedirs(common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + 'rx/' + 'after_adapt/' + str(num_pilots_test) + '_num_pilots_test/')
            os.makedirs(common_dir + 'test_result/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/'  + str(num_pilots_test) + '_num_pilots_test/')
            test_result_per_num_pilots_test = common_dir + 'test_result/'  + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + str(num_pilots_test) + '_num_pilots_test/'+ 'test_result.mat'
            save_test_result_dict_per_num_pilots_test = {}

            block_error_rate = torch.zeros(args.num_channels_test, len(test_snr_range))
            ind_h = 0
            for h in h_list_test:
                print('current channel ind', ind_h)
                PATH_after_adapt_tx = common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + 'tx/' + 'after_adapt/' + str(num_pilots_test) + '_num_pilots_test/' + str(ind_h) + 'th_adapted_net'
                PATH_after_adapt_rx = common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/'  + 'rx/' + 'after_adapt/' + str(
                    num_pilots_test) + '_num_pilots_test/' + str(ind_h) + 'th_adapted_net'
                test_training(args, h, tx_net_for_testtraining, rx_net_for_testtraining, Noise, Noise_relax, PATH_before_adapt_tx, PATH_before_adapt_rx, PATH_after_adapt_tx, PATH_after_adapt_rx,num_pilots_test)
                # test
                ind_snr = 0
                for test_snr in test_snr_range:
                    block_error_rate_per_snr_per_channel = test_per_channel_per_snr(args, args.test_size, h, tx_net_for_testtraining, rx_net_for_testtraining, test_snr, actual_channel_num, PATH_after_adapt_tx, PATH_after_adapt_rx)
                    block_error_rate[ind_h, ind_snr] = block_error_rate_per_snr_per_channel
                    total_block_error_rate[ind_h, ind_snr, ind_num_pilots_test] = block_error_rate_per_snr_per_channel
                    total_total_block_error_rate[ind_h, ind_snr, ind_num_pilots_test, ind_meta_training_epoch] = block_error_rate_per_snr_per_channel
                    ind_snr += 1
                ind_h += 1
            save_test_result_dict_per_num_pilots_test['block_error_rate'] = block_error_rate.detach().numpy()
            sio.savemat(test_result_per_num_pilots_test, save_test_result_dict_per_num_pilots_test)
            writer_test.add_scalar('average (h) block error rate per num pilots', torch.mean(block_error_rate[:, :]), num_pilots_test)
            ind_num_pilots_test += 1
            print('curr pilots num', num_pilots_test, 'bler', torch.mean(block_error_rate[:, :]))

        save_test_result_dict['block_error_rate_total'] = total_block_error_rate.detach().numpy()

        if common_dir_over_multi_rand_seeds is not None:
            sio.savemat(test_result_all_PATH_per_rand_seeds, save_test_result_dict)
        else:
            test_result_all_PATH = common_dir + 'test_result/' + 'with_meta_training_epoch/' + str(
                meta_training_epochs) + '/' + 'test_result.mat'

            sio.savemat(test_result_all_PATH, save_test_result_dict)

def test_with_adapt_compact_during_online_meta_training(args, common_dir, curr_meta_training_epoch, test_snr_range, num_pilots_test, tx_net_for_testtraining, rx_net_for_testtraining, Noise, Noise_relax, actual_channel_num, PATH_before_adapt_tx, PATH_before_adapt_rx):
    # generate or load test channels
    if args.path_for_test_channels is None:
        raise NotImplementedError
    else:
        print('load previously generated channels')
        h_list_test_path = args.path_for_test_channels + '/' + 'test_channels.pckl'
        f_test_channels = open(h_list_test_path, 'rb')
        h_list_test = pickle.load(f_test_channels)
        f_test_channels.close()

    if len(h_list_test) > args.num_channels_test:
        h_list_test = h_list_test[:args.num_channels_test]

    # reset again # to make fair comp. per adapt. and per meta-training epochs
    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 2)
    print('curr pilots used for test during online (meta-)learning: ', num_pilots_test)

    os.makedirs(common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(
        curr_meta_training_epoch) + '/' + 'tx/' + 'after_adapt/' + str(num_pilots_test) + '_num_pilots_test/')
    os.makedirs(common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(
        curr_meta_training_epoch) + '/' + 'rx/' + 'after_adapt/' + str(num_pilots_test) + '_num_pilots_test/')

    block_error_rate = torch.zeros(args.num_channels_test, len(test_snr_range))
    ind_h = 0
    for h in h_list_test:
        PATH_after_adapt_tx = common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(curr_meta_training_epoch) + '/' + 'tx/' + 'after_adapt/' + str(num_pilots_test) + '_num_pilots_test/'+ str(ind_h) + 'th_adapted_net'
        PATH_after_adapt_rx = common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(curr_meta_training_epoch) + '/'  + 'rx/' + 'after_adapt/' + str(
            num_pilots_test) + '_num_pilots_test/' + str(ind_h) + 'th_adapted_net'
        test_training(args, h, tx_net_for_testtraining, rx_net_for_testtraining, Noise, Noise_relax, PATH_before_adapt_tx, PATH_before_adapt_rx, PATH_after_adapt_tx, PATH_after_adapt_rx, num_pilots_test)
        # test
        ind_snr = 0
        for test_snr in test_snr_range:
            block_error_rate_per_snr_per_channel = test_per_channel_per_snr(args, args.test_size_during_meta_update, h, tx_net_for_testtraining, rx_net_for_testtraining, test_snr, actual_channel_num, PATH_after_adapt_tx, PATH_after_adapt_rx)
            block_error_rate[ind_h, ind_snr] = block_error_rate_per_snr_per_channel
            ind_snr += 1
        ind_h += 1
    return torch.mean(block_error_rate[:, :])

def test_conven_commun_during_online_meta_training(args, test_snr_range, num_pilots_test, Noise, actual_channel_num):
    # generate or load test channels
    if args.path_for_test_channels is None:
        print('we need to first make the test channels a priori and load it via path (args.path_for_test_channels)')
        raise NotImplementedError
    else:
        print('load previously generated channels')
        h_list_test_path = args.path_for_test_channels + '/' + 'test_channels.pckl'
        f_test_channels = open(h_list_test_path, 'rb')
        h_list_test = pickle.load(f_test_channels)
        f_test_channels.close()

    if len(h_list_test) > args.num_channels_test:
        h_list_test = h_list_test[:args.num_channels_test]
    print('used test channels', h_list_test)

    # reset again # to make fair comp. per adapt. and per meta-training epochs
    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 2)
    print('curr num pilots: ', num_pilots_test)

    block_error_rate = torch.zeros(args.num_channels_test, len(test_snr_range))
    ind_h = 0
    ch_est_error_avg = 0
    for h in h_list_test:
        est_h, error_h = test_training_conven_commun(args, h, Noise, num_pilots_test)
        ind_snr = 0
        for test_snr in test_snr_range:
            block_error_rate_per_snr_per_channel = test_per_channel_per_snr_conven_approach(args, est_h, args.conv_payload_num, h, test_snr, actual_channel_num)
            block_error_rate[ind_h, ind_snr] = block_error_rate_per_snr_per_channel
            ind_snr += 1
        ind_h += 1
        ch_est_error_avg += error_h
    ch_est_error_avg = ch_est_error_avg/ind_h

    return torch.mean(block_error_rate[:, :]), ch_est_error_avg

