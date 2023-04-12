#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=75000
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --job-name hist2st-pf
#SBATCH --array=0-9
#module load cuda/11.0.2.450
#module load gnu7
#module load openmpi3
module load anaconda/3.6
source activate ../envs/Hist2ST

SRC_DIR=$(cd ../envs/Hist2ST/bin; pwd)
PATH=${SRC_DIR}:$PATH

cwd=$(pwd)

cd ../models/Hist2ST

CUDA_LAUNCH_BLOCKING=1 python ${cwd}/Hist2ST_pf.py $SLURM_ARRAY_TASK_ID

# NO POINT, BECAUSE MODEL DIFFERENT (NO 1000 GENE model)
# CUDA_LAUNCH_BLOCKING=1 python ${cwd}/Hist2ST_pf-pretrained.py $SLURM_ARRAY_TASK_ID

