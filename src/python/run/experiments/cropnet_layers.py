# How many layers do we need to label the grid accurately?
from run.training.cropnet.train import train

baseline = [{'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1}]

for i in range(7):
    architecture = baseline.copy()
    for j in range(i):
        architecture.append(
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1})
    train(architecture=architecture, work_dir='cropnet52x52-{}layers'.format(i + 3))
