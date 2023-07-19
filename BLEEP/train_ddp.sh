#!/bin/bash
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --array=0-8

MY_DIR=/scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/models/BLEEP/
cd $MY_DIR

module load anaconda/3.6
source activate /scratch/imb/uqjxie6/software/stimage5
# module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2
module load git-2.19.1-gcc-7.3.0-swjt5hp

export PATH=/scratch/imb/uqjxie6/software/stimage5:$PATH
export PATH=/scratch/imb/uqjxie6/software/stimage5/bin:$PATH

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"


# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

# srun python main.py --init_method tcp://$MASTER_ADDR:3456 --exp_name clip_a1 --world_size $SLURM_NTASKS  --batch_size 256 --max_epochs 300 --num_workers 4
# srun python $1 --init_method tcp://$MASTER_ADDR:3456 --exp_name $2 --world_size $SLURM_NTASKS  --batch_size $3 --max_epochs $4 --num_workers $5

python BLEEP_main.py --init_method tcp://$MASTER_ADDR:3456 --exp_name clip_visium_bc --batch_size 256 --max_epochs 150 --num_workers 4 --fold $SLURM_ARRAY_TASK_ID --dim 5689 --data_name visium_bc --model CLIPModel_cosine_sim

#--model CLIPModel_cosine_sim, CLIPModel_VICReg
#sbatch train_ddp_slice1.sh main.py clip_a1 256 300 4


