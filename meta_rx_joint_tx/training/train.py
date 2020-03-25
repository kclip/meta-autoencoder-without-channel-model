import torch
from utils.funcs import one_iter_sim_fix_stoch_encoder, one_iter_sim_fix_stoch_encoder_from_rx_meta_tx_joint

def test_training(args, h, tx_net_for_testtraining, rx_net_for_testtraining, Noise, Noise_relax, Noise_feedback, PATH_before_adapt_tx, PATH_before_adapt_rx, PATH_after_adapt_tx, PATH_after_adapt_rx, adapt_steps, writer_per_test_channel_tx, writer_per_test_channel_rx): #PATH_before_adapt can be meta-learneds
    # initialize network (net_for_testtraining) (net is for meta-training)
    tx_net_for_testtraining.load_state_dict(torch.load(PATH_before_adapt_tx))
    rx_net_for_testtraining.load_state_dict(torch.load(PATH_before_adapt_rx))

    if args.if_test_training_adam and not args.if_adam_after_sgd: #if if_adam_after_sgd it will generated after one sgd step
        tx_testtraining_optimiser = torch.optim.Adam(tx_net_for_testtraining.parameters(), args.lr_testtraining)
        rx_testtraining_optimiser = torch.optim.Adam(rx_net_for_testtraining.parameters(), args.lr_testtraining)
    else:
        tx_testtraining_optimiser = None
        rx_testtraining_optimiser = None

    num_adapt = adapt_steps # considering 10 tx & 10 rx as 1 epoch
    for epochs in range(num_adapt):
        if args.tx_rx_training_mode == 0:
            if args.if_joint_training_tx_meta_training_rx:
                rx_testtraining_optimiser, tx_testtraining_optimiser, loss_curr_rx, loss_curr_tx = one_iter_sim_fix_stoch_encoder_from_rx_meta_tx_joint(
                    args, h, Noise, Noise_relax, Noise_feedback, tx_net_for_testtraining,
                    rx_net_for_testtraining,
                    tx_testtraining_optimiser, rx_testtraining_optimiser,
                    epochs)
            else:
                rx_testtraining_optimiser, tx_testtraining_optimiser, loss_curr_rx, loss_curr_tx = one_iter_sim_fix_stoch_encoder(args, h, Noise, Noise_relax, Noise_feedback, tx_net_for_testtraining,
                                                                rx_net_for_testtraining,
                                                                tx_testtraining_optimiser, rx_testtraining_optimiser,
                                                                epochs)
        else:
            print('we are considering only simultaneous training')
            raise NotImplementedError

    torch.save(tx_net_for_testtraining.state_dict(), PATH_after_adapt_tx)
    torch.save(rx_net_for_testtraining.state_dict(), PATH_after_adapt_rx)
