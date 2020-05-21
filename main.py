import torch
import argparse
from nets.transmitter import tx_dnn
from nets.receiver import rx_dnn
from training.meta_train import multi_task_learning
import scipy.io as sio
import datetime
import numpy
import os
from utils.basic_funcs import reset_randomness
from utils.test_bler_with_adaptation import test_with_adapt


def parse_args():
    parser = argparse.ArgumentParser(description='end_to_end-meta')
    parser.add_argument('--bit_num', type=int, default=8, help='number of bits')
    parser.add_argument('--channel_num', type=int, default=4, help='number of channel uses')
    parser.add_argument('--Eb_over_N_db', type=float, default=15, help='energy per bit to noise power spectral density ratio')
    parser.add_argument('--Eb_over_N_db_test', type=float, default=15, help='Eb/N0 in db for test')
    parser.add_argument('--num_neurons_encoder', type=int, default=None, help='number of neuron in hidden layer in encoder')
    parser.add_argument('--num_neurons_decoder', type=int, default=None, help='number of neuron in hidden layer in decoder')
    parser.add_argument('--if_cpu', dest='if_cpu', action='store_true', default=False)
    parser.add_argument('--if_bias', dest='if_bias', action='store_true', default=False)
    parser.add_argument('--if_relu', dest='if_relu', action='store_true', default=False)
    parser.add_argument('--cuda_ind', type=int, default=0, help='index for cuda device')
    parser.add_argument('--lr_testtraining', type=float, default=0.001, help='simple sgd')
    parser.add_argument('--lr_meta_update', type=float, default=0.01, help='lr for meta-update')
    parser.add_argument('--lr_meta_inner', type=float, default=0.1, help='lr for meta-inner')
    parser.add_argument('--test_size', type=int, default=1000000, help='number of transmission blocks used to compute BLER after meta-training')

    parser.add_argument('--path_for_common_dir', dest='path_for_common_dir',
                        default='default_folder/default_subfolder/', type=str)
    parser.add_argument('--path_for_common_dir_only_for_test_results', dest='path_for_common_dir_only_for_test_results',
                        default=None, type=str)
    parser.add_argument('--tap_num', type=int, default=3, help='number of taps')
    parser.add_argument('--num_channels_test', type=int, default=100, help='number of new channels to get BLER (working as test channels)')
    parser.add_argument('--path_for_test_channels', dest='path_for_test_channels', default=None, type=str)
    parser.add_argument('--path_for_meta_trained_net_rx', dest='path_for_meta_trained_net_rx', default=None, type=str)
    parser.add_argument('--path_for_meta_trained_net_tx', dest='path_for_meta_trained_net_tx', default=None, type=str)
    parser.add_argument('--tasks_per_metaupdate', type=int, default=1, help='mini batch size of tasks (channels)')
    parser.add_argument('--num_meta_local_updates', type=int, default=1, help='number of local updates in meta-training') # number of local pingpongs
    parser.add_argument('--num_epochs_meta_train', type=int, default=10000,
                        help='number epochs for meta-training')
    parser.add_argument('--if_RTN', dest='if_RTN', action='store_true', default=False, help='if use RTN layer for the receiver')
    parser.add_argument('--inv_filter_num_added', type=int, default=0,
                        help='number added to tap number to define output size of RTN layer')
    parser.add_argument('--if_joint_training', dest='if_joint_training', action='store_true', default=False, help='if joint training') # else: meta-learning
    parser.add_argument('--if_test_training_adam', dest='if_test_training_adam', action='store_true',
                        default=False, help='if use ADAM optimizer for gradient updates during run time')
    parser.add_argument('--if_adam_after_sgd', dest='if_adam_after_sgd', action='store_true',
                        default=False, help='if first use SGD then ADAM for gradient updates during run time') # desinged for MAML to use SGD during args.num_meta_local_updates
    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--if_fix_random_seed', dest='if_fix_random_seed', action='store_true',
                        default=False, help='if fix random seed')

    parser.add_argument('--spawc_setting', dest='spawc_setting', action='store_true',
                        default=False, help='settings for SPAWC paper')

    parser.add_argument('--relax_sigma', type=float, default=0.15,
                        help='standard deviation for relaxation used in stochatic encoder')
    parser.add_argument('--if_AWGN', action='store_true',
                        default=False, help='consider AWGN instead of rayleigh block fading')

    parser.add_argument('--if_joint_training_tx_meta_training_rx', action='store_true',
                        default=False, help='proposed hybrid scheme')

    parser.add_argument('--if_Rayleigh_channel_model_AR', action='store_true',
                        default=False, help='correlated channel model')

    parser.add_argument('--rho', type=float, default=0.99,
                        help='correlation coefficient between previous channel')

    parser.add_argument('--if_online_meta_training', action='store_true',
                        default=False, help='if online meta-learning, should be true always')

    parser.add_argument('--see_test_bler_during_meta_update', action='store_true',
                        default=False, help='whether to see BLER of new channels while meta-training (e.g., for Fig. 3)')

    parser.add_argument('--test_size_during_meta_update', type=int, default=100000, help='number of transmission blocks used to compute BLER during meta-training')

    parser.add_argument('--meta_tr_epoch_num_for_test', type=int, default=10000,
                        help='how often do we compute BLER of new channels during meta-training')

    parser.add_argument('--if_fully_random_no_fixed_seed_highest_priority', action='store_true',
                        default=False, help='if we want to run without fixing random seed')

    parser.add_argument('--if_TB_loss_ignore', action='store_true',
                        default=False, help='if we want to ignore tensorboard visualizing losses during meta-training')

    parser.add_argument('--if_get_conven_commun_performance', action='store_true',
                        default=False, help='if we want to get conventional communication performance (BPSK + maximum likelihood decoder)')
    parser.add_argument('--conv_payload_num', type=int, default=100,
                        help='number of transmission blocks for test')
    parser.add_argument('--if_use_stopping_criteria_during_meta_training', action='store_true',
                        default=False, help='works as early stopping during meta-training based on losses during meta-training')

    parser.add_argument('--num_pilots_in_frame_meta_test', type=int, default=None,
                        help='number of pilots used for meta-test performance during online training') # unless specified, set as default
    parser.add_argument('--num_pilots_in_frame_meta_train', type=int, default=None,
                        help='number of whole transmission blocks in frame during online (meta-)training') # unless specified, set as default
    parser.add_argument('--num_pilots_meta_train_used_for_support', type=int, default=None,
                        help='number of pilots in frame (T_U in paper) during online (meta-)training') # unless specified, set as default
    parser.add_argument('--separate_meta_training_support_query_set', action='store_true',
                        default=False, help='if we want to separate transmission blocks used for local adaptation (support set) apart from transmission blocks used for computing meta-gradient (query set)')
    parser.add_argument('--fix_tx_multi_adapt_rx_iter_num', type=int, default=1,
                        help='number of adaptation at run time for reciever')

    parser.add_argument('--fix_bpsk_tx_train_nn_rx_during_runtime', action='store_true',
                        default=False, help='when we want to consider BPSK encoder with NN decoder to be adapted during runtime')
    parser.add_argument('--fix_bpsk_tx', action='store_true',
                        default=False, help='when we want to (meta-)train NN decoder under BPSK encoder via online learning')
    parser.add_argument('--if_exp_over_multi_pilots_test', action='store_true',
                        default=False, help='if we want to run experiments over varying number of pilots during test (during runtime, e.g., Fig. 4)')


    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.cuda_ind) if torch.cuda.is_available() else "cpu")

    if args.num_neurons_encoder == None:
        args.num_neurons_encoder = pow(2,args.bit_num)
    if args.num_neurons_decoder == None:
        args.num_neurons_decoder = pow(2, args.bit_num)

    if args.if_test_training_adam == False:
        args.if_adam_after_sgd = False
    print('Running on device: {}'.format(args.device))
    if args.if_cpu:
        args.device = torch.device("cpu")

    if args.spawc_setting == True:
        print('running for a more realistic scenario (first case)')
        args.Eb_over_N_db = 10
        args.Eb_over_N_db_test = 10
        args.bit_num = 8
        args.channel_num = 8
        args.tap_num = 3
        args.pilots_num_meta_test = 256
        args.pilots_num_meta_train_supp = 8
        args.pilots_num_meta_train_query = 256
        args.num_neurons_encoder = 16
        args.num_neurons_decoder = 16
        args.if_fix_random_seed = True  # for reproducibility and fair comparison of randomness
        args.if_online_meta_training = True
    else:
        print('running on custom environment')

    if args.if_online_meta_training:
        args.if_Rayleigh_channel_model_AR = True
        args.tasks_per_metaupdate = 1
        args.see_test_bler_during_meta_update = True

    if args.if_fully_random_no_fixed_seed_highest_priority:
        args.if_fix_random_seed = False

    if args.if_fix_random_seed:
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        numpy.random.seed(args.random_seed)

    if args.num_pilots_in_frame_meta_test is not None:
        args.pilots_num_meta_test = args.num_pilots_in_frame_meta_test
    if args.num_pilots_in_frame_meta_train is not None:
        args.pilots_num_meta_train_query = args.num_pilots_in_frame_meta_train
    if args.num_pilots_meta_train_used_for_support is not None:
        args.pilots_num_meta_train_supp = args.num_pilots_meta_train_used_for_support
    if args.num_epochs_meta_train == 0:
        args.if_use_stopping_criteria_during_meta_training = False
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    common_dir = './' + args.path_for_common_dir + curr_time + '/'

    if args.path_for_common_dir_only_for_test_results is not None:
        common_dir_over_multi_rand_seeds = './' + args.path_for_common_dir_only_for_test_results  + '/'
    else:
        common_dir_over_multi_rand_seeds = None

    PATH_before_adapt_rx = common_dir + 'saved_model/' + 'rx/' + 'before_adapt/' + 'init_net'
    PATH_before_adapt_tx = common_dir + 'saved_model/' + 'tx/' + 'before_adapt/' + 'init_net'

    PATH_before_adapt_rx_intermediate = common_dir + 'saved_model/' + 'rx/' + 'during_meta_training/' + 'epochs/'
    PATH_before_adapt_tx_intermediate = common_dir + 'saved_model/' + 'tx/' + 'during_meta_training/' + 'epochs/'

    os.makedirs(common_dir + 'saved_model/' + 'rx/' + 'before_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'tx/' + 'before_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'rx/' + 'after_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'tx/' + 'after_adapt/')
    os.makedirs(PATH_before_adapt_rx_intermediate)
    os.makedirs(PATH_before_adapt_tx_intermediate)
    os.makedirs(common_dir + 'test_channels/')
    os.makedirs(common_dir + 'test_result/')
    os.makedirs(common_dir + 'test_result_during_meta_training/')

    dir_meta_training = common_dir + 'TB/' + 'meta_training'
    writer_meta_training = SummaryWriter(dir_meta_training)

    dir_during_adapt = common_dir + 'TB/' + 'during_adapt/'


    meta_training_epoch_for_test = [args.num_epochs_meta_train]
    save_test_result_dict_total = {}
    test_result_all_PATH_for_all_meta_training_epochs = None

    test_snr_range = [args.Eb_over_N_db_test]

    if args.if_exp_over_multi_pilots_test:
        test_num_pilots_available = [1,2,4,8,16,32,64,128]
    else:
        test_num_pilots_available = [8]
    print('test available pilots: ', test_num_pilots_available)

    test_result_PATH_per_meta_training_test_bler = common_dir + 'test_result_during_meta_training/' + 'test_result_per_meta_training_epochs.mat'

    save_test_result_dict_total_per_meta_training_test_bler = {}

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

    # reset random seed again
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


    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 10)
    else:
        pass
    if (args.path_for_meta_trained_net_tx is not None) or (args.path_for_meta_trained_net_rx is not None):
        PATH_before_adapt_rx = args.path_for_meta_trained_net_rx
        PATH_before_adapt_tx = args.path_for_meta_trained_net_tx
    else:
        if args.if_joint_training:
            print('start joint training')
        else:
            print('start meta-training')
        pass

        test_bler_per_meta_training_epochs, channel_per_meta_training_epochs = multi_task_learning(args, common_dir, tx_net, rx_net, writer_meta_training, Noise, Noise_relax, actual_channel_num, PATH_before_adapt_rx_intermediate, PATH_before_adapt_tx_intermediate, rx_net_for_testtraining, tx_net_for_testtraining)

        torch.save(rx_net.state_dict(), PATH_before_adapt_rx)
        torch.save(tx_net.state_dict(), PATH_before_adapt_tx)

        save_test_result_dict_total_per_meta_training_test_bler[
            'test_bler_during_meta_training'] = test_bler_per_meta_training_epochs
        save_test_result_dict_total_per_meta_training_test_bler[
            'channel_during_meta_training'] = channel_per_meta_training_epochs
        sio.savemat(test_result_PATH_per_meta_training_test_bler, save_test_result_dict_total_per_meta_training_test_bler)
        if args.path_for_common_dir_only_for_test_results is not None:
            test_result_PATH_per_meta_training_test_bler_over_rand_seeds = common_dir_over_multi_rand_seeds + 'test_result_during_meta_training/' + 'rho/' + str(args.rho) + '/rand_seeds/' + str(args.random_seed) + '/' +  'test_result_per_meta_training_epochs.mat'
            os.makedirs(common_dir_over_multi_rand_seeds + 'test_result_during_meta_training/' + 'rho/' + str(args.rho) + '/rand_seeds/' + str(args.random_seed))
            sio.savemat(test_result_PATH_per_meta_training_test_bler_over_rand_seeds,
                        save_test_result_dict_total_per_meta_training_test_bler)
        else:
            pass


        if args.if_use_stopping_criteria_during_meta_training:
            PATH_before_adapt_rx = PATH_before_adapt_rx_intermediate + 'best_model_based_on_meta_training_loss'
            PATH_before_adapt_tx = PATH_before_adapt_tx_intermediate + 'best_model_based_on_meta_training_loss'
        else:
            pass

            #############
    test_with_adapt(args, common_dir, common_dir_over_multi_rand_seeds, test_snr_range, test_num_pilots_available, meta_training_epoch_for_test,
                    tx_net_for_testtraining,
                    rx_net_for_testtraining, Noise, Noise_relax,
                    actual_channel_num, save_test_result_dict_total, test_result_all_PATH_for_all_meta_training_epochs, PATH_before_adapt_tx, PATH_before_adapt_rx)