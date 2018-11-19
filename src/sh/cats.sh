#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/nfs/pdurnay
#SBATCH --gres=gpu:pascal:1
#SBATCH --export=ALL
srun python3 dronevision/src/python/trainings/cats.py