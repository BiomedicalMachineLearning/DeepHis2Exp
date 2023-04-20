#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=75000
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --job-name stnet-pf-cv
#SBATCH --array=0-9

module load cuda/11.0.2.450
module load gnu7
module load openmpi3
module load anaconda/3.6

source activate ../envs/stimage_test

PATH=../envs/stimage_test/bin:$PATH

python ./stnet_pf-cv-224.py $SLURM_ARRAY_TASK_ID
