#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=75000
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --job-name stimage-pf
#SBATCH --array=0

module load cuda/11.0.2.450
module load gnu7
module load openmpi3
module load anaconda/3.6

source activate ../envs/stimage_test

PATH=../envs/stimage_test/bin:$PATH

cd ../models/STimage/stimage/

time python 01_Preprocessing.py --config /scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/data/pfizer2/stimage-config.ini

time python 02_Training_DUMMY.py --config /scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/data/pfizer2/stimage-config.ini
# time python 02_Training.py --config /scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/data/pfizer/stimage-config.ini

# time python 03_Prediction.py --config /scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/data/pfizer/stimage-config.ini
  
# Pretty sure this does nothing
# time python 04_Interpretation.py --config /scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/data/pfizer/stimage-config.ini
 
# time python 05_Visualisation.py --config /scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/data/pfizer/stimage-config.ini
# time python 05_Visualisation_MOD.py --config /scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/data/pfizer/stimage-config.ini

