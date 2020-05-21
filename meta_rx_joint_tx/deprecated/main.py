import torch
import argparse
from data_gen.data_set import channel_set_gen
from training.train import test_training
from training.test import test_per_channel_per_snr
from nets.transmitter import tx_dnn
from nets.receiver import rx_dnn
from torch.utils.tensorboard import SummaryWriter
from training.meta_train import multi_task_learning
import pickle
import scipy.io as sio
import datetime
import numpy
import os
from utils.basic_funcs import reset_randomness

def parse_args():
    parser = argparse.ArgumentParser(description='end_to_end-meta')
    parser.add_argument('--bit_num', type=int, default=8, help='number of bits')
    parser.add_argument('--channel_num', type=int, default=4, help='number of channel uses')
    parser.add_argument('--mb_size', type=int, default=256, help='minibatch size')
    parser.add_argument('--Eb_over_N_db', type=float, default=15, help='energy per bit to noise power spectral density ratio')
    parser.add_argument('--Eb_over_N_db_test', type=float, default=15, help='Eb/N0 in db for test')
    parser.add_argument('--num_neurons_encoder', type=int, default=None, help='number of neuron in hidden layer in encoder')
    parser.add_argument('--num_neurons_decoder', type=int, default=None, help='number of neuron in hidden layer in decoder')
    parser.add_argument('--if_cpu', dest='if_cpu', action='store_true', default=False)
    parser.add_argument('--if_bias', dest='if_bias', action='store_true', default=False)
    parser.add_argument('--if_relu', dest='if_relu', action='store_true', default=False)
    parser.add_argument('--cuda_ind', type=int, default=0, help='index for cuda device')
    parser.add_argument('--lr_testtraining', type=float, default=0.1, help='simple sgd')
    parser.add_argument('--lr_meta_update', type=float, default=0.01, help='lr for meta-update')
    parser.add_argument('--lr_meta_inner', type=float, default=0.1, help='lr for meta-inner')
    parser.add_argument('--test_size', type=int, default=1000000, help='..')
    parser.add_argument('--path_for_common_dir', dest='path_for_common_dir',
                        default='default_folder/default_subfolder/', type=str)
    parser.add_argument('--tap_num', type=int, default=3, help='..')
    parser.add_argument('--num_channels_meta', type=int, default=100, help='..')
    parser.add_argument('--num_channels_test', type=int, default=20, help='..')
    parser.add_argument('--path_for_meta_training_channels', dest='path_for_meta_training_channels', default=None, type=str)
    parser.add_argument('--path_for_test_channels', dest='path_for_test_channels', default=None, type=str)
    parser.add_argument('--path_for_meta_trained_net_rx', dest='path_for_meta_trained_net_rx', default=None, type=str)
    parser.add_argument('--path_for_meta_trained_net_tx', dest='path_for_meta_trained_net_tx', default=None, type=str)
    parser.add_argument('--tasks_per_metaupdate', type=int, default=20, help='mini batch size of tasks (channels)')
    parser.add_argument('--num_meta_local_updates', type=int, default=1, help='number of local updates in meta-training') # number of local pingpongs
    parser.add_argument('--mb_size_meta_train', type=int, default=16, help='minibatch size during meta-training (this can be useful for decreasing pilots)')
    parser.add_argument('--mb_size_meta_test', type=int, default=16,
                        help='minibatch size for query set (this can be useful for decreasing pilots)')
    parser.add_argument('--num_epochs_meta_train', type=int, default=10000,
                        help='number epochs for meta-training')
    parser.add_argument('--if_RTN', dest='if_RTN', action='store_true', default=False)
    parser.add_argument('--inv_filter_num_added', type=int, default=0,
                        help='number added to tap number')
    parser.add_argument('--if_joint_training', dest='if_joint_training', action='store_true', default=False) # else: meta-learning for multi-task learning
    parser.add_argument('--if_test_training_adam', dest='if_test_training_adam', action='store_true',
                        default=False)
    parser.add_argument('--if_adam_after_sgd', dest='if_adam_after_sgd', action='store_true',
                        default=False) # desinged for maml sgd during args.num_meta_local_updates
    parser.add_argument('--random_seed', type=int, default=0,
                        help='...')
    parser.add_argument('--if_fix_random_seed', dest='if_fix_random_seed', action='store_true',
                        default=False)
    ### RL
    parser.add_argument('--relax_sigma', type=float, default=0.15,
                        help='...')
    parser.add_argument('--if_AWGN', action='store_true',
                        default=False) # AWGN channel
    parser.add_argument('--tx_rx_training_mode', type=int, default=0,
                        help='...') # 0: simulatneous stoch. autoencoder (currently, only this mode)
    parser.add_argument('--mul_test_adapt_range', type=float, default=None,
                        help='...')
    parser.add_argument('--if_calc_meta_exact_explicily', action='store_true',
                        default=False)
    parser.add_argument('--noisy_feedback_snr_db_meta_train', type=float, default=0,
                        help='dB scale')
    parser.add_argument('--noisy_feedback_snr_db_meta_test', type=float, default=0,
                        help='dB scale')
    parser.add_argument('--if_noisy_feedback_during_meta_train', action='store_true',
                        default=False)
    parser.add_argument('--if_noisy_feedback_during_meta_test', action='store_true',
                        default=False)
    parser.add_argument('--path_for_meta_trained_net_total_per_epoch', dest='path_for_meta_trained_net_total_per_epoch', default=None, type=str)
    parser.add_argument('--if_always_generate_new_meta_training_channels', action='store_true',
                        default=False)
    parser.add_argument('--if_joint_training_tx_meta_training_rx', action='store_true',
                        default=False)
    parser.add_argument('--if_realistic_scenario', action='store_true',
                        default=False)
    parser.add_argument('--fix_joint_trained_tx_only_adapt_meta_trained_rx', action='store_true',
                        default=False) # currently, only for meta-trained autoencoder
    parser.add_argument('--specific_meta_training_epoch_for_test', type=int, default=None,
                        help='...')
    # AR model for Rayleigh channel
    parser.add_argument('--if_Rayleigh_channel_model_AR', action='store_true',
                        default=False)
    parser.add_argument('--rho', type=float, default=0.99,
                        help='...')
    parser.add_argument('--keep_AR_period', type=int, default=100000,
                        help='...')
    parser.add_argument('--mul_h_var_min', type=int, default=1,
                        help='...')
    parser.add_argument('--mul_h_var_max', type=int, default=1,
                        help='...')


    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.cuda_ind) if torch.cuda.is_available() else "cpu")

    assert args.if_always_generate_new_meta_training_channels == args.if_Rayleigh_channel_model_AR

    if args.if_always_generate_new_meta_training_channels:
        args.num_channels_meta = 1 # we are not using saved meta-trainig channels, this is dummy variable

    if args.num_neurons_encoder == None:
        args.num_neurons_encoder = pow(2,args.bit_num)
    if args.num_neurons_decoder == None:
        args.num_neurons_decoder = pow(2, args.bit_num)

    if args.if_test_training_adam == False:
        args.if_adam_after_sgd = False

    print('Running on device: {}'.format(args.device))
    if args.if_cpu:
        args.device = torch.device("cpu")
    if args.if_realistic_scenario == True: # used pre-defined setting
        args.tasks_per_metaupdate = 1 # practical
        args.test_size = 10000
        args.num_channels_test = 20
        args.if_bias = True
        args.if_relu = True
        args.if_RTN = True
        args.Eb_over_N_db = 10
        args.Eb_over_N_db_test = 10
        args.bit_num = 8
        args.channel_num = 3
        args.tap_num = 3
        args.mb_size = 256
        args.mb_size_meta_train = 256
        args.mb_size_meta_test = 256
        args.num_channels_meta = 1
        args.num_neurons_encoder = 16
        args.num_neurons_decoder = 16
        args.if_fix_random_seed = True  # for reproducibility and fair comparison of randomness
        args.if_test_training_adam = True
        args.if_always_generate_new_meta_training_channels = True
        args.if_Rayleigh_channel_model_AR = True
        args.if_adam_after_sgd = True
    else:
        print('running on custom environment')

    if args.if_Rayleigh_channel_model_AR:
        assert args.tasks_per_metaupdate == 1 # currently...

    if args.if_fix_random_seed:
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        numpy.random.seed(args.random_seed)
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    common_dir = '/home/hdd1/logs/meta_rx_joint_tx/' + args.path_for_common_dir + curr_time + '/'

    PATH_before_adapt_rx = common_dir + 'saved_model/' + 'rx/' + 'before_adapt/' + 'init_net'
    PATH_before_adapt_tx = common_dir + 'saved_model/' + 'tx/' + 'before_adapt/' + 'init_net'

    PATH_before_adapt_rx_intermediate = common_dir + 'saved_model/' + 'rx/' + 'during_meta_training/' + 'epochs/'
    PATH_before_adapt_tx_intermediate = common_dir + 'saved_model/' + 'tx/' + 'during_meta_training/' + 'epochs/'

    os.makedirs(common_dir + 'saved_model/' + 'rx/' + 'before_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'tx/' + 'before_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'tx_critic/' + 'before_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'rx/' + 'after_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'tx/' + 'after_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'tx_critic/' + 'after_adapt/')
    os.makedirs(PATH_before_adapt_rx_intermediate)
    os.makedirs(PATH_before_adapt_tx_intermediate)

    os.makedirs(common_dir + 'meta_training_channels/')
    os.makedirs(common_dir + 'test_channels/')
    os.makedirs(common_dir + 'test_result/')

    dir_meta_training = common_dir + 'TB/' + 'meta_training'
    writer_meta_training = SummaryWriter(dir_meta_training)
    dir_during_adapt = common_dir + 'TB/' + 'during_adapt/'

    if args.path_for_meta_trained_net_total_per_epoch is not None: # is not None, already net. given in the regarding dir.
        assert args.path_for_meta_trained_net_rx == None
        assert args.path_for_meta_trained_net_tx == None
        # get max epoch of current path
        saved_meta_training_epochs_list = os.listdir(args.path_for_meta_trained_net_total_per_epoch + '/rx/during_meta_training/epochs')
        saved_meta_training_epochs_list = list(map(int, saved_meta_training_epochs_list))
        possible_max_saved_meta_training_epoch = max(saved_meta_training_epochs_list)
        # if performance check with more than this epoch, use last saved epochs

    if args.path_for_meta_trained_net_total_per_epoch:
        meta_training_epoch_for_test = [1000,2000,5000,10000,20000,30000,40000,50000]
        if args.if_joint_training: # for fair comp.
            meta_training_epoch_for_test =  [int(i * 2) for i in meta_training_epoch_for_test]
        test_result_all_PATH_for_all_meta_training_epochs = common_dir + 'test_result/' + 'test_result_all_epochs.mat'
        save_test_result_dict_total = {}
    else:
        meta_training_epoch_for_test = [args.num_epochs_meta_train]
        save_test_result_dict_total = {}

    test_snr_range = [args.Eb_over_N_db_test]  # [15]
    test_adapt_range = [0, 1, 2, 3, 4, 5]
    print('test adapt range: ', test_adapt_range)
    # complex symbol
    actual_channel_num = args.channel_num * 2

    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 999)

    tx_net = tx_dnn(M=pow(2, args.bit_num), num_neurons_encoder=args.num_neurons_encoder, n=actual_channel_num, if_bias=args.if_bias,
              if_relu=args.if_relu)

    if torch.cuda.is_available():
        tx_net = tx_net.to(args.device)

    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 999)

    tx_net_for_testtraining = tx_dnn(M=pow(2, args.bit_num), num_neurons_encoder=args.num_neurons_encoder,
                                     n=actual_channel_num,
                                     if_bias=args.if_bias,
                                     if_relu=args.if_relu)

    if torch.cuda.is_available():
        tx_net_for_testtraining = tx_net_for_testtraining.to(args.device)

    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 99)

    rx_net = rx_dnn(M=pow(2, args.bit_num), n=actual_channel_num,
              n_inv_filter=args.tap_num + args.inv_filter_num_added,
              num_neurons_decoder=args.num_neurons_decoder, if_bias=args.if_bias,
              if_relu=args.if_relu, if_RTN=args.if_RTN, if_fix_random_seed = args.if_fix_random_seed, random_seed = args.random_seed + 99)

    if torch.cuda.is_available():
        rx_net = rx_net.to(args.device)

    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 99)

    rx_net_for_testtraining = rx_dnn(M=pow(2, args.bit_num), n=actual_channel_num,
                    n_inv_filter=args.tap_num + args.inv_filter_num_added,
                    num_neurons_decoder=args.num_neurons_decoder, if_bias=args.if_bias,
                    if_relu=args.if_relu, if_RTN=args.if_RTN, if_fix_random_seed = args.if_fix_random_seed, random_seed = args.random_seed + 99)

    if torch.cuda.is_available():
        rx_net_for_testtraining = rx_net_for_testtraining.to(args.device)

    # reset again
    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 1)

    Eb_over_N = pow(10, (args.Eb_over_N_db/10))
    R = args.bit_num/args.channel_num
    noise_var = 1 / (2 * R * Eb_over_N) # real and imaginary
    Noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(actual_channel_num), noise_var * torch.eye(actual_channel_num))

    # relaxation distribution
    if args.relax_sigma > 0:
        Noise_relax = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(actual_channel_num),
                                                                           pow(args.relax_sigma, 2) * torch.eye(actual_channel_num))
    else:
        Noise_relax = None
    # noise of feedback link during meta-training
    noisy_feedback_snr_meta_train = pow(10, (args.noisy_feedback_snr_db_meta_train / 10))
    noise_feedback_var_meta_train = 1/noisy_feedback_snr_meta_train # power 1 as coincides with power constraint of tx (do not need to be in this way though)

    if args.if_noisy_feedback_during_meta_train:
        Noise_feedback_meta_train = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(1),
                                                                                      noise_feedback_var_meta_train * torch.eye(1))
    else:
        Noise_feedback_meta_train = None
    # noise of feedback link during meta-test
    noisy_feedback_snr_meta_test = pow(10, (args.noisy_feedback_snr_db_meta_test / 10))
    noise_feedback_var_meta_test = 1 / noisy_feedback_snr_meta_test  # power 1 as coincides with power constraint of tx (do not need to be in this way though)

    if args.if_noisy_feedback_during_meta_test:
        Noise_feedback_meta_test = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(1),
                                                                                               noise_feedback_var_meta_test * torch.eye(
                                                                                                   1))
    else:
        Noise_feedback_meta_test = None

    if args.path_for_meta_training_channels is None:
        if args.if_fix_random_seed:
            reset_randomness(args.random_seed + 9)
        #print('generate meta-training channels')
        h_list_meta = channel_set_gen(args.num_channels_meta, args.tap_num)
        #print('meta-training channels: ', h_list_meta)
        h_list_meta_path = common_dir + 'meta_training_channels/' + 'training_channels.pckl'
        f_meta_channels = open(h_list_meta_path, 'wb')
        pickle.dump(h_list_meta, f_meta_channels)
        f_meta_channels.close()
    else:
        print('load previously generated channels')
        h_list_meta_path = args.path_for_meta_training_channels + '/' + 'training_channels.pckl'
        f_meta_channels = open(h_list_meta_path, 'rb')
        h_list_meta = pickle.load(f_meta_channels)
        print('meta-training channels: ', h_list_meta)
        f_meta_channels.close()

    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 10)

    if args.path_for_meta_trained_net_total_per_epoch is not None:
        pass
        print('load meta-trained network per meta-training epoch with values: ', meta_training_epoch_for_test)
    elif (args.path_for_meta_trained_net_tx is not None) and (args.path_for_meta_trained_net_rx is not None):
        PATH_before_adapt_rx = args.path_for_meta_trained_net_rx
        PATH_before_adapt_tx = args.path_for_meta_trained_net_tx
    else:
        if args.if_joint_training:
            print('start joint training')
        else:
            print('start meta-training')
        pass
        multi_task_learning(args, tx_net, rx_net, h_list_meta, writer_meta_training, Noise, Noise_relax, Noise_feedback_meta_train, PATH_before_adapt_rx_intermediate, PATH_before_adapt_tx_intermediate)

        torch.save(rx_net.state_dict(), PATH_before_adapt_rx)
        torch.save(tx_net.state_dict(), PATH_before_adapt_tx)
    if args.path_for_test_channels is None:
        if args.if_fix_random_seed:
            reset_randomness(args.random_seed + 11)
        print('generate test channels')
        h_list_test = channel_set_gen(args.num_channels_test, args.tap_num)
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

    ## per meta-trained network per meta-training epoch
    total_total_block_error_rate = torch.zeros(args.num_channels_test, len(test_snr_range), len(test_adapt_range), len(meta_training_epoch_for_test))
    ind_meta_training_epoch = 0
    for meta_training_epochs in meta_training_epoch_for_test:
        if args.path_for_meta_trained_net_total_per_epoch is not None:
            print(meta_training_epochs)
            print(possible_max_saved_meta_training_epoch)
            if meta_training_epochs <= possible_max_saved_meta_training_epoch:
                print('test with meta-training epoch ended at: ', meta_training_epochs)
                PATH_before_adapt_rx = args.path_for_meta_trained_net_total_per_epoch + '/rx/during_meta_training/epochs/' + str(meta_training_epochs)
                PATH_before_adapt_tx = args.path_for_meta_trained_net_total_per_epoch + '/tx/during_meta_training/epochs/' + str(meta_training_epochs)
            else:
                print('do not have such epochs for current meta-trained net, so use final saved network instead')
                PATH_before_adapt_rx = args.path_for_meta_trained_net_total_per_epoch + '/rx/before_adapt/init_net'
                PATH_before_adapt_tx = args.path_for_meta_trained_net_total_per_epoch + '/tx/before_adapt/init_net'

        else:
            pass
        os.makedirs(common_dir + 'test_result/' + 'with_meta_training_epoch/' +  str(meta_training_epochs) + '/' )
        test_result_all_PATH = common_dir + 'test_result/' + 'with_meta_training_epoch/' +  str(meta_training_epochs) + '/' + 'test_result.mat'

        save_test_result_dict = {}
        # start with given initialization
        print('start adaptation with test set')
        total_block_error_rate = torch.zeros(args.num_channels_test, len(test_snr_range), len(test_adapt_range))
        ind_adapt_steps = 0
        for adapt_steps in test_adapt_range:
            # reset again # to make fair comp. per adapt. and per meta-training epochs
            if args.if_fix_random_seed:
                reset_randomness(args.random_seed + 2)
            print('curr adaptation: ', adapt_steps)
            os.makedirs(common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + 'tx/' + 'after_adapt/' + str(adapt_steps) + '_adapt_steps/')
            os.makedirs(common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + 'tx_critic/' + 'after_adapt/' + str(adapt_steps) + '_adapt_steps/')
            os.makedirs(common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + 'rx/' + 'after_adapt/' + str(adapt_steps) + '_adapt_steps/')
            os.makedirs(common_dir + 'test_result/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/'  + str(adapt_steps) + '_adapt_steps/')
            test_result_per_adapt_steps = common_dir + 'test_result/'  + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + str(adapt_steps) + '_adapt_steps/' + 'test_result.mat'
            save_test_result_dict_per_adapt_steps = {}

            block_error_rate = torch.zeros(args.num_channels_test, len(test_snr_range))
            ind_h = 0
            for h in h_list_test:
                print('current channel ind', ind_h)
                PATH_after_adapt_tx = common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + 'tx/' + 'after_adapt/' + str(adapt_steps) + '_adapt_steps/'+ str(ind_h) + 'th_adapted_net'
                PATH_after_adapt_tx_critic = common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/' + 'tx_critic/' + 'after_adapt/' + str(adapt_steps) + '_adapt_steps/'+ str(ind_h) + 'th_adapted_net'
                PATH_after_adapt_rx = common_dir + 'saved_model/' + 'with_meta_training_epoch/' + str(meta_training_epochs) + '/'  + 'rx/' + 'after_adapt/' + str(
                    adapt_steps) + '_adapt_steps/' + str(ind_h) + 'th_adapted_net'
                writer_per_test_channel_tx = None
                writer_per_test_channel_rx = None
                test_training(args, h, tx_net_for_testtraining, rx_net_for_testtraining, Noise, Noise_relax, Noise_feedback_meta_test, PATH_before_adapt_tx, PATH_before_adapt_rx, PATH_after_adapt_tx, PATH_after_adapt_rx,adapt_steps, writer_per_test_channel_tx, writer_per_test_channel_rx)
                # test
                ind_snr = 0
                for test_snr in test_snr_range:
                    block_error_rate_per_snr_per_channel = test_per_channel_per_snr(args, h, tx_net_for_testtraining, rx_net_for_testtraining, test_snr, actual_channel_num, PATH_after_adapt_tx, PATH_after_adapt_rx, writer_per_test_channel_tx, writer_per_test_channel_rx)
                    block_error_rate[ind_h, ind_snr] = block_error_rate_per_snr_per_channel
                    total_block_error_rate[ind_h, ind_snr, ind_adapt_steps] = block_error_rate_per_snr_per_channel
                    total_total_block_error_rate[ind_h, ind_snr, ind_adapt_steps, ind_meta_training_epoch] = block_error_rate_per_snr_per_channel
                    ind_snr += 1
                ind_h += 1
            ind_snr = 0
            save_test_result_dict_per_adapt_steps['block_error_rate'] = block_error_rate.detach().numpy()
            sio.savemat(test_result_per_adapt_steps, save_test_result_dict_per_adapt_steps)
            writer_test.add_scalar('average (h) block error rate per adaptation steps', torch.mean(block_error_rate[:, :]), adapt_steps)
            ind_adapt_steps += 1

        ind_meta_training_epoch += 1

        save_test_result_dict['block_error_rate_total'] = total_block_error_rate.detach().numpy()
        sio.savemat(test_result_all_PATH, save_test_result_dict)

    if args.path_for_meta_trained_net_total_per_epoch:
        save_test_result_dict_total[
            'block_error_rate_total_total_meta_training_epoch'] = total_total_block_error_rate.detach().numpy()
        sio.savemat(test_result_all_PATH_for_all_meta_training_epochs, save_test_result_dict_total)