#!/bin/bash

python BLEEP_main.py --init_method tcp://$MASTER_ADDR:3456 --exp_name clip_visium_bc_holdout_alex --batch_size 256 --max_epochs 150 --num_workers 4 --dim 5689 --data_name visium_bc --holdout 1


