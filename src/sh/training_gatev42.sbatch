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
srun python3 dronevision/src/python/run/training/gate/train.py --model_name GateNetV42 --work_dir gatev42 --img_res 52,52
