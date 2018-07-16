#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=32:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/nfs/pdurnay
#SBATCH --gres=gpu:pascal:1
#SBATCH --export=ALL
module use /opt/insy/modulefiles
module load cuda/9.0 cudnn/9.0-7.0.5
srun python3 dronevision/src/python/run/experiments/1807/bottleneck.py
