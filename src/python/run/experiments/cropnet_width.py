# How many filters per layer do we need to label the grid accurately?
from run.training.cropnet.train import train

for i in range(3):
    architecture = [{'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16 * i, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16 * i, 'strides': (1, 1), 'alpha': 0.1}]

    train(architecture=architecture, work_dir='cropnet52x52-3layers-16x{}kernel'.format(i))

for i in range(3):
    architecture = [{'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16 * i, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16 * i, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16 * i, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16 * i, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16 * i, 'strides': (1, 1), 'alpha': 0.1}]

    train(architecture=architecture, work_dir='cropnet52x52-6layers-16x{}kernel'.format(i))
