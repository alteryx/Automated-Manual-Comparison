#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --time=6000
#SBATCH --mail-user=wjk68@case.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name='Random Search'

module load StdEnv
module load python/3.5.1
cd /mnt/projects/CSE_MSE_RXF131/hpc-members/wjk68/git/modeling/scripts
python random_search.py