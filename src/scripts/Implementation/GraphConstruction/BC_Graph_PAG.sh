#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name GNN_BC
#SBATCH --array=0-8

module load cuda/11.0.2.450
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /scratch/imb/uqyjia11/.conda/envs/pt3.8

PATH=/scratch/imb/uqyjia11/.conda/envs/pt3.8/bin:$PATH
python ./BuildGraph_dgl_PAG.py --fold $SLURM_ARRAY_TASK_ID 
