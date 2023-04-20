#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=75000
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --job-name stimage-pf-cv
#SBATCH --array=0

module load cuda/11.0.2.450
module load gnu7
module load openmpi3
module load anaconda/3.6

source activate ../envs/stimage_test

PATH=../envs/stimage_test/bin:$PATH

# python ./stimage_pf_cv.py $SLURM_ARRAY_TASK_ID

python ./stimage_testt.py $SLURM_ARRAY_TASK_ID
