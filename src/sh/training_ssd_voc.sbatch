#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/nfs/pdurnay
#SBATCH --gres=gpu:pascal:1
#SBATCH --export=ALL
srun python3 dronevision/src/python/run/training/voc/ssd.py
