#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=32:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/nfs/pdurnay
#SBATCH --gres=gpu:pascal:1
#SBATCH --export=ALL
srun lib/darknet detector train $HOME/dronevision/resource/ext/samples/gates-mixed-yolo20k/gates-mixed-yolo20k.txt $HOME/dronevision/lib/darknet/cfg/yolov2-gatenet10.cfg
