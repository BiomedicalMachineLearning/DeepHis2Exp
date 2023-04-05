#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=75000
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH --job-name deepspace
#SBATCH --gres=gpu:tesla-smx2:1

cd ~/scratch/benchmmarking/DeepHis2Exp/
cd models/DeepSpaCE/

module load anaconda/3.6
module load cuda/10.2.89.440
module load gnu/5.4.0
module load mvapich2
module load git-2.19.1-gcc-7.3.0-swjt5hp
module load singularity/3.4.1

export SINGULARITY_CACHEDIR=$HOME/scratch/.singularity
export SINGULARITY_TMPDIR=$SINGULARITY_CACHEDIR/tmp
export SINGULARITY_LOCALCACHEDIR=$SINGULARITY_CACHEDIR/localcache
export SINGULARITY_PULLFOLDER=$SINGULARITY_CACHEDIR/pull

chmod +x train-pf.sh
singularity exec --nv deepspace_v1.0.sif train-pf.sh 
