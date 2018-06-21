# How many layers do we need to label the grid accurately?
import argparse

from run.training.cropnet.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--w", help="Image Width",
                    type=int, default=52)
parser.add_argument("--h", help="Image Height",
                    type=int, default=52)

args = parser.parse_args()
h, w = args.h, args.h
baseline = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)}]

for i in range(1, 8):
    architecture = baseline.copy()

    for j in range(i - 1):
        architecture.append(
            {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1})
        if len(architecture) == 3:
            architecture.append({'name': 'max_pool', 'size': (2, 2)})

    if len(architecture) == 2:
        architecture.append({'name': 'max_pool', 'size': (2, 2)})

    architecture.append(
        {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 4, 'strides': (1, 1), 'alpha': 0.1})
    train(architecture=architecture, work_dir='cropnet{}x{}-{}layers'.format(h, w, i + 1), epochs=50, img_res=(h, w))
