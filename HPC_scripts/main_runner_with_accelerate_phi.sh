#!/bin/sh
#SBATCH --job-name=Phi_Finetune
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node040
#SBATCH --time=12:00:00
#SBATCH --output /work/%u/VLA/results/phi/job%j.out
#SBATCH --error /work/%u/VLA/results/phi/job%j.err
#SBATCH -p gpu-H200


# Load CUDA module
module load cuda12.4
user=$(whoami)
# Load Anaconda module (adjust the path if necessary)
module load python3/anaconda/3.12

# Activate your conda environment, ## user source activate on cluster, not conda activate
source activate /work/$user/VLA/phi_env
cd activate /work/$user/VLA
# Add this line to pass all the arguments down to main.py
accelerate launch --config_file ./Config/phi_ds_config.yaml /work/$user/VLA/main.py "$@"

