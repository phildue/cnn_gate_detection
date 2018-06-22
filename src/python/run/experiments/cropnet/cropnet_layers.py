# How many layers do we need to label the grid accurately?
import argparse

from run.training.cropnet.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--w", help="Image Width",
                    type=int, default=52)
parser.add_argument("--h", help="Image Height",
                    type=int, default=52)
parser.add_argument("--x_filter_start", help="16*x_filter per layer start",
                    type=int, default=1)
parser.add_argument("--x_filter_end", help="16*x_filter per layer end",
                    type=int, default=1)

args = parser.parse_args()
h, w = args.h, args.h
for x in range(args.x_filter_start, args.x_filter_end):
    baseline = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16 * x, 'strides': (1, 1), 'alpha': 0.1},
                {'name': 'max_pool', 'size': (2, 2)}]

    for i in range(1, 8):
        architecture = baseline.copy()

        for j in range(i - 1):
            architecture.append(
                {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16 * x, 'strides': (1, 1), 'alpha': 0.1})
            if len(architecture) == 3:
                architecture.append({'name': 'max_pool', 'size': (2, 2)})

        if len(architecture) == 2:
            architecture.append({'name': 'max_pool', 'size': (2, 2)})

        architecture.append(
            {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 4, 'strides': (1, 1), 'alpha': 0.1})
        train(architecture=architecture, work_dir='cropnet{}x{}-{}layers-{}filters'.format(h, w, i + 1, 16 * x),
              epochs=50, img_res=(h, w))
