# How many layers do we need for a 52x52 crop?
import argparse

from run.training.refine.train import train


def run(baseline, name, start_layers, n_layers, width):
    for i in range(1, n_layers):
        architecture = baseline.copy()

        for j in range(i - 1):
            architecture.append(
                {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': width, 'strides': (1, 1), 'alpha': 0.1})

        train(architecture=architecture, work_dir='{}+{}layers+{}filters'.format(name, start_layers + i, width),
              img_res=(52, 52),
              epochs=50)


parser = argparse.ArgumentParser()
parser.add_argument("--x_filter_start", help="16*x_filter per layer start",
                    type=int, default=1)
parser.add_argument("--x_filter_end", help="16*x_filter per layer end",
                    type=int, default=1)

args = parser.parse_args()
for x in range(args.x_filter_start, args.x_filter_end):
    width = 16 * x
    baseline13x13 = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                     {'name': 'max_pool', 'size': (2, 2)},
                     {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                     {'name': 'max_pool', 'size': (2, 2)}
                     ]
    baseline6x6 = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                   {'name': 'max_pool', 'size': (2, 2)},
                   {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                   {'name': 'max_pool', 'size': (2, 2)},
                   {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                   {'name': 'max_pool', 'size': (2, 2)}
                   ]
    baseline3x3 = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                   {'name': 'max_pool', 'size': (2, 2)},
                   {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                   {'name': 'max_pool', 'size': (2, 2)},
                   {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                   {'name': 'max_pool', 'size': (2, 2)},
                   {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                   {'name': 'max_pool', 'size': (2, 2)}
                   ]

    run(baseline13x13, 'refnet52x52-13x13', 2, 7, width)
    run(baseline6x6, 'refnet52x52-6x6', 3, 6, width)
    run(baseline6x6, 'refnet52x52-3x3', 4, 5, width)
