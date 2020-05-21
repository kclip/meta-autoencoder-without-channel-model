import torch
import numpy as np
from numpy.linalg import inv
from utils.basic_funcs import complex_mul_taps


def four_qam_uncoded_modulation(num_bits, num_complex_ch_uses, label):
    if num_bits == num_complex_ch_uses:
        bpsk = [-1, 1]
        if num_bits == 8:
            actual_transmitted_symbol = torch.zeros(label.shape[0], 16)
            for ind_mb in range(label.shape[0]):
                ind_message = label[ind_mb]  # 0,1,...,15
                ind_message_binary = bin(ind_message)[2:]

                if len(ind_message_binary) < 8:
                    while len(ind_message_binary) < 8:
                        ind_message_binary = '0' + ind_message_binary
                actual_transmitted_symbol[ind_mb, 0] = bpsk[int(ind_message_binary[0])]
                actual_transmitted_symbol[ind_mb, 2] = bpsk[int(ind_message_binary[1])]
                actual_transmitted_symbol[ind_mb, 4] = bpsk[int(ind_message_binary[2])]
                actual_transmitted_symbol[ind_mb, 6] = bpsk[int(ind_message_binary[3])]
                actual_transmitted_symbol[ind_mb, 8] = bpsk[int(ind_message_binary[4])]
                actual_transmitted_symbol[ind_mb, 10] = bpsk[int(ind_message_binary[5])]
                actual_transmitted_symbol[ind_mb, 12] = bpsk[int(ind_message_binary[6])]
                actual_transmitted_symbol[ind_mb, 14] = bpsk[int(ind_message_binary[7])]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return actual_transmitted_symbol


def channel_estimation(args, actual_transmitted_symbol_torch, received_signal, num_taps):
    if num_taps == 3:
        actual_transmitted_symbol = actual_transmitted_symbol_torch.numpy()

        if args.bit_num == 8:
            A = np.zeros((8 * actual_transmitted_symbol.shape[0], 3), dtype=complex)

            for ind_mb in range(actual_transmitted_symbol.shape[0]):
                A[8 * ind_mb, 0] = actual_transmitted_symbol[ind_mb, 0] + actual_transmitted_symbol[ind_mb, 1] * 1j  # x_1
                A[8 * ind_mb, 1] = 0
                A[8 * ind_mb, 2] = 0

                A[8 * ind_mb + 1, 0] = actual_transmitted_symbol[ind_mb, 2] + actual_transmitted_symbol[ind_mb, 3] * 1j  # x_2
                A[8 * ind_mb + 1, 1] = actual_transmitted_symbol[ind_mb, 0] + actual_transmitted_symbol[ind_mb, 1] * 1j  # x_1
                A[8 * ind_mb + 1, 2] = 0

                A[8 * ind_mb + 2, 0] = actual_transmitted_symbol[ind_mb, 4] + actual_transmitted_symbol[ind_mb, 5] * 1j  # x_3
                A[8 * ind_mb + 2, 1] = actual_transmitted_symbol[ind_mb, 2] + actual_transmitted_symbol[ind_mb, 3] * 1j  # x_2
                A[8 * ind_mb + 2, 2] = actual_transmitted_symbol[ind_mb, 0] + actual_transmitted_symbol[ind_mb, 1] * 1j  # x_1

                A[8 * ind_mb + 3, 0] = actual_transmitted_symbol[ind_mb, 6] + actual_transmitted_symbol[ind_mb, 7] * 1j  # x_4
                A[8 * ind_mb + 3, 1] = actual_transmitted_symbol[ind_mb, 4] + actual_transmitted_symbol[ind_mb, 5] * 1j  # x_3
                A[8 * ind_mb + 3, 2] = actual_transmitted_symbol[ind_mb, 2] + actual_transmitted_symbol[ind_mb, 3] * 1j  # x_2

                A[8 * ind_mb + 4, 0] = actual_transmitted_symbol[ind_mb, 8] + actual_transmitted_symbol[ind_mb, 9] * 1j  # x_5
                A[8 * ind_mb + 4, 1] = actual_transmitted_symbol[ind_mb, 6] + actual_transmitted_symbol[ind_mb, 7] * 1j  # x_4
                A[8 * ind_mb + 4, 2] = actual_transmitted_symbol[ind_mb, 4] + actual_transmitted_symbol[ind_mb, 5] * 1j  # x_3

                A[8 * ind_mb + 5, 0] = actual_transmitted_symbol[ind_mb, 10] + actual_transmitted_symbol[ind_mb, 11] * 1j  # x_6
                A[8 * ind_mb + 5, 1] = actual_transmitted_symbol[ind_mb, 8] + actual_transmitted_symbol[ind_mb, 9] * 1j  # x_5
                A[8 * ind_mb + 5, 2] = actual_transmitted_symbol[ind_mb, 6] + actual_transmitted_symbol[ind_mb, 7] * 1j  # x_4

                A[8 * ind_mb + 6, 0] = actual_transmitted_symbol[ind_mb, 12] + actual_transmitted_symbol[ind_mb, 13] * 1j  # x_7
                A[8 * ind_mb + 6, 1] = actual_transmitted_symbol[ind_mb, 10] + actual_transmitted_symbol[ind_mb, 11] * 1j  # x_6
                A[8 * ind_mb + 6, 2] = actual_transmitted_symbol[ind_mb, 8] + actual_transmitted_symbol[ind_mb, 9] * 1j  # x_5

                A[8 * ind_mb + 7, 0] = actual_transmitted_symbol[ind_mb, 14] + actual_transmitted_symbol[ind_mb, 15] * 1j  # x_8
                A[8 * ind_mb + 7, 1] = actual_transmitted_symbol[ind_mb, 12] + actual_transmitted_symbol[ind_mb, 13] * 1j  # x_7
                A[8 * ind_mb + 7, 2] = actual_transmitted_symbol[ind_mb, 10] + actual_transmitted_symbol[ind_mb, 11] * 1j  # x_6

            received_signal_total = np.zeros((8 * actual_transmitted_symbol.shape[0], 1), dtype=complex)
            for ind_mb in range(actual_transmitted_symbol.shape[0]):
                received_signal_curr_mb_in_real_format = received_signal[ind_mb].cpu().numpy()
                received_signal_curr_mb = np.zeros((8, 1), dtype=complex)
                for ind_y in range(8):
                    received_signal_curr_mb[ind_y] = received_signal_curr_mb_in_real_format[2 * ind_y] + \
                                                     received_signal_curr_mb_in_real_format[2 * ind_y + 1] * 1j
                    received_signal_total[ind_mb * 8 + ind_y] = received_signal_curr_mb[ind_y]

            I = np.zeros((8 * actual_transmitted_symbol.shape[0], 8 * actual_transmitted_symbol.shape[0]), dtype=complex)
            for i in range(8 * actual_transmitted_symbol.shape[0]):
                I[i, i] = 1
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    ## mmse estimator
    ch_var_complex = 1 / num_taps

    Eb_over_N = pow(10, (args.Eb_over_N_db / 10))
    R = args.bit_num / args.channel_num
    noise_var_complex = 1 / (R * Eb_over_N)

    cov_h_y = ch_var_complex * np.conj(np.transpose(A))

    cov_y = ch_var_complex * np.matmul(A, np.conj(np.transpose(A))) + noise_var_complex * I

    W = np.matmul(cov_h_y, inv(cov_y))
    estimated_ch_avg = np.matmul(W, received_signal_total)
    return estimated_ch_avg


def demodulation(num_bits, num_ch, num_taps, estimated_ch, received_signal_test, device):
    # maximum likelihood
    out = torch.zeros(received_signal_test.shape[0], pow(2,num_bits))
    for ind_mb in range(received_signal_test.shape[0]):
        # h_complex to h_real 2d
        est_ch_tensor = torch.zeros(2*num_taps).to(device) # tap = 3
        for ind_h in range(num_taps):
            est_ch_tensor[2*ind_h] = np.real(estimated_ch[ind_h])[0]
            est_ch_tensor[2*ind_h+1] = np.imag(estimated_ch[ind_h])[0]

        for cand_m_ind in range(pow(2,num_bits)):
            label = torch.zeros(1,dtype=torch.int)
            label[0] = int(cand_m_ind)
            cand_x = four_qam_uncoded_modulation(num_bits, num_ch, label)
            cand_y = complex_mul_taps(est_ch_tensor, cand_x)
            out[ind_mb, cand_m_ind] = 1 / torch.norm(received_signal_test[ind_mb] - cand_y.to(device)) # since norm smaller, better
    return out.to(device)