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
srun python src\python\run\training\gate\train.py --model_name gatev26 --learning_rate 0.0001 --initial_epoch 40 --epochs 50 --model_src out/gatev26/