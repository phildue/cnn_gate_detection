#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/nfs/pdurnay
#SBATCH --gres=gpu:pascal:1
#SBATCH --export=ALL
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev5 logs/gatev5_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev6 logs/gatev6_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev7 logs/gatev7_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev8 logs/gatev8_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev9 logs/gatev9_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev10 logs/gatev10_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev11 logs/gatev11_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev12 logs/gatev12_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py gatev13 logs/gatev13_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py tiny_yolo logs/tiny_mixed/
srun python3 dronevision/scripts/python/evaluation/evalspeed.py yolo logs/v2_mixed/
