#!/bin/bash

python BLEEP_main.py --init_method tcp://$MASTER_ADDR:3456 --exp_name clip_visium_bc --batch_size 256 --max_epochs 150 --num_workers 4 --fold $SLURM_ARRAY_TASK_ID --dim 5689 --data_name visium_bc
