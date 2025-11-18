#!/bin/sh
#SBATCH --job-name=VLMFinetune
#SBATCH -N 1    ## requests on 1 node
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output /work/%u/VLA/results/job%j.out
#SBATCH --error /work/%u/VLA/results/job%j.err
#SBATCH -p gpu-H200


# Load CUDA module
module load cuda12.4
user=$(whoami)
# Load Anaconda module (adjust the path if necessary)
module load python3/anaconda/3.12


# Activate your conda environment, ## user source activate on cluster, not conda activate
source activate /work/$user/VLA/qwen_env

# Add this line to pass all the arguments down to main.py
python /work/$user/VLA/main.py "$@"

