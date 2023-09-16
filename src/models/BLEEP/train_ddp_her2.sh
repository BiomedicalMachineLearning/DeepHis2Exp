#!/bin/bash

python BLEEP_main.py --init_method tcp://$MASTER_ADDR:3456 --exp_name clip_her2_subset --batch_size 256 --max_epochs 150 --num_workers 4 --fold $SLURM_ARRAY_TASK_ID --dim 769 --data_name her2_subset
