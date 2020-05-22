import torch
from utils.funcs import one_frame_conventional_training_tx_bpsk_rx_nn, one_frame_conventional_training_tx_nn_rx_nn, one_iter_mmse_ch_est

def test_training(args, h, tx_net_for_testtraining, rx_net_for_testtraining, Noise, Noise_relax, PATH_before_adapt_tx, PATH_before_adapt_rx, PATH_after_adapt_tx, PATH_after_adapt_rx, num_pilots_test): #PATH_before_adapt can be meta-learneds
    # initialize network (net_for_testtraining) (net is for meta-training)
    if args.fix_bpsk_tx_train_nn_rx_during_runtime:
        pass
    else:
        tx_net_for_testtraining.load_state_dict(torch.load(PATH_before_adapt_tx))
    rx_net_for_testtraining.load_state_dict(torch.load(PATH_before_adapt_rx))

    tx_testtraining_optimiser = None # since we do not have feedback link during runtime

    if args.if_test_training_adam and not args.if_adam_after_sgd:  # if if_adam_after_sgd it will generated after one sgd step
        rx_testtraining_optimiser = torch.optim.Adam(rx_net_for_testtraining.parameters(), args.lr_testtraining)
    else:
        rx_testtraining_optimiser = None

    if num_pilots_test == 0:
        num_adapt = 0
        num_pilots_test_in_one_mb = 0
    else: # we use all possible pilots at once to update the neural networks
        num_adapt = 1
        num_pilots_test_in_one_mb = num_pilots_test

    for epochs in range(num_adapt):
        if args.fix_bpsk_tx_train_nn_rx_during_runtime:
            rx_testtraining_optimiser, loss_curr_rx, loss_curr_tx = one_frame_conventional_training_tx_bpsk_rx_nn(
                args, h, Noise,
                rx_net_for_testtraining,
                rx_testtraining_optimiser,
                num_pilots_test_in_one_mb)
        elif args.if_fix_nn_tx_train_nn_rx_during_runtime:
            rx_testtraining_optimiser, loss_curr_rx, loss_curr_tx = one_frame_conventional_training_tx_nn_rx_nn(
                args, h, Noise, Noise_relax, tx_net_for_testtraining,
                rx_net_for_testtraining, rx_testtraining_optimiser,
                epochs, num_pilots_test_in_one_mb)
        else:
            raise NotImplementedError # deprecated...

    torch.save(tx_net_for_testtraining.state_dict(), PATH_after_adapt_tx)
    torch.save(rx_net_for_testtraining.state_dict(), PATH_after_adapt_rx)


def test_training_conven_commun(args, h, Noise, num_pilots_test):
    # initialize network (net_for_testtraining) (net is for meta-training)
    est_h_avg = None
    num_adapt = 1
    print('num pilots for ch est', num_pilots_test)
    for epochs in range(num_adapt):
        est_h, error_h = one_iter_mmse_ch_est(args, h, Noise, num_pilots_test)
        if epochs == 0:
            est_h_avg = est_h
        else:
            est_h_avg += est_h
    est_h_avg = est_h_avg/num_adapt
    return est_h_avg, error_h


