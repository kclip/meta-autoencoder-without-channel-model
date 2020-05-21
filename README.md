## Meta-Autoencoder without Channel Model

This repository contains code for "[End-to-End Fast Training of Communication Links Without a Channel Model via Online Meta-Learning](https://arxiv.org/abs/2003.01479)" - 
Sangwoo Park, Osvaldo Simeone, and Joonhyuk Kang.

### Dependencies

This program is written in python 3.7 and uses PyTorch 1.2.0 and scipy.
Tensorboard for pytorch is used for visualization (e.g., https://pytorch.org/docs/stable/tensorboard.html).
- pip install tensorboard and pip install scipy might be useful.

### Basic Usage under Rayleigh Block Fading channel case
    
-  In the 'run/' folder, all the schemes in the paper can be trained via correlated channel model and tested with the same channels as done in the paper.

### Preliminary codes are now deprecated
    
-  Codes in the folder 'meta_rx_joint_tx/', which was preliminary version of the current code is now deprecated and will be erased soon.
