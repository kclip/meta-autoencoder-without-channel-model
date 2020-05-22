import torch
from data_gen.data_set import message_gen
from utils.funcs import channel
from utils.basic_funcs import reset_randomness
from utils.conv_communication_scheme import four_qam_uncoded_modulation, demodulation

def test_per_channel_per_snr(args, test_size, h, tx_net_for_testtraining, rx_net_for_testtraining, test_snr, actual_channel_num, PATH_after_adapt_tx, PATH_after_adapt_rx):

    tx_net_for_testtraining.load_state_dict(torch.load(PATH_after_adapt_tx))
    rx_net_for_testtraining.load_state_dict(torch.load(PATH_after_adapt_rx))

    batch_size = test_size
    success_test = 0
    Eb_over_N_test = pow(10, (test_snr / 10))
    R = args.bit_num / args.channel_num
    noise_var_test = 1 / (2 * R * Eb_over_N_test)
    Noise_test = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(actual_channel_num),
                                                                            noise_var_test * torch.eye(actual_channel_num))
    m_test, label_test = message_gen(args.bit_num, batch_size)
    m_test = m_test.type(torch.FloatTensor).to(args.device)
    label_test = label_test.type(torch.LongTensor).to(args.device)

    for f in tx_net_for_testtraining.parameters():
        if f.grad is not None:
            f.grad.detach()
            f.grad.zero_()

    rx_net_for_testtraining.zero_grad()

    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 7777) # always have same noise for test since this can see the actual effect of # of adaptations


    if args.fix_bpsk_tx_train_nn_rx_during_runtime:
        actual_transmitted_symbol = four_qam_uncoded_modulation(args.bit_num, args.channel_num,
                                                                label_test)  # label instead m
    else:
        tx_symb_mean, actual_transmitted_symbol = tx_net_for_testtraining(m_test, args.device, 0, None) # no relaxation
    tx_symb_mean = None # we don't need during test
    # channel
    received_signal = channel(h, actual_transmitted_symbol, Noise_test, args.device, args.if_AWGN)
    # rx
    out_test = rx_net_for_testtraining(received_signal, args.if_RTN, args.device)

    for ind_mb in range(label_test.shape[0]):
        assert label_test.shape[0] == batch_size
        if torch.argmax(out_test[ind_mb]) == label_test[ind_mb]:  # means correct classification
            success_test += 1
        else:
            pass
    accuracy = success_test / label_test.shape[0]

    return 1 - accuracy


def test_per_channel_per_snr_conven_approach(args, est_h, test_size, h, test_snr, actual_channel_num):
    batch_size = test_size
    success_test = 0
    Eb_over_N_test = pow(10, (test_snr / 10))
    R = args.bit_num / args.channel_num
    noise_var_test = 1 / (2 * R * Eb_over_N_test)
    Noise_test = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(actual_channel_num),
                                                                            noise_var_test * torch.eye(actual_channel_num))
    print('payload num for conv', batch_size)
    m_test, label_test = message_gen(args.bit_num, batch_size)
    label_test = label_test.type(torch.LongTensor).to(args.device)

    if args.if_fix_random_seed:
        reset_randomness(args.random_seed + 7777) # always have same noise for test since this can see the actual effect of # of adaptations

    actual_transmitted_symbol = four_qam_uncoded_modulation(args.bit_num, args.channel_num, label_test)#m_test)

    received_signal = channel(h, actual_transmitted_symbol, Noise_test, args.device, args.if_AWGN)
    # rx (maximum likelihood)
    out_test = demodulation(args.bit_num, args.channel_num, args.tap_num, est_h, received_signal, args.device)
    for ind_mb in range(label_test.shape[0]):
        assert label_test.shape[0] == batch_size
        if torch.argmax(out_test[ind_mb]) == label_test[ind_mb]:  # means correct classification
            success_test += 1
        else:
            pass
    accuracy = success_test / label_test.shape[0]

    print('accuracy', accuracy)

    return 1 - accuracy
