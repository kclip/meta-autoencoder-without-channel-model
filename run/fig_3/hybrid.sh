python ../../main.py --meta_tr_epoch_num_for_test 100 --num_epochs_meta_train 60000 --if_bias --if_RTN --if_relu --spawc_setting --if_fix_random_seed  --fix_tx_multi_adapt_rx_iter_num 1 --if_joint_training_tx_meta_training_rx --if_fix_nn_tx_train_nn_rx_during_runtime --if_online_meta_training --rho 0.5 --random_seed 0 --if_use_stopping_criteria_during_meta_training --num_pilots_in_frame_meta_test 8 --separate_meta_training_support_query_set --lr_testtraining 0.1 --lr_meta_inner 0.1  --path_for_common_dir 'fig_3/hybrid/' --path_for_test_channels '../../test_channels'