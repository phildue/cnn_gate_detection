# How many layers do we need for a 52x52 crop?
from run.training.refine.train import train


def run(baseline, name, start_layers, n_layers):
    for i in range(1, n_layers):
        architecture = baseline.copy()

        for j in range(i - 1):
            architecture.append(
                {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1})

        train(architecture=architecture, work_dir='{}+{}layers'.format(name, start_layers + i), img_res=(52, 52))


baseline13x13 = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                 {'name': 'max_pool', 'size': (2, 2)},
                 {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                 {'name': 'max_pool', 'size': (2, 2)}
                 ]
baseline6x6 = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
               {'name': 'max_pool', 'size': (2, 2)},
               {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
               {'name': 'max_pool', 'size': (2, 2)},
               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
               {'name': 'max_pool', 'size': (2, 2)}
               ]
baseline3x3 = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
               {'name': 'max_pool', 'size': (2, 2)},
               {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
               {'name': 'max_pool', 'size': (2, 2)},
               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
               {'name': 'max_pool', 'size': (2, 2)},
               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
               {'name': 'max_pool', 'size': (2, 2)}
               ]

run(baseline13x13, 'refnet52x52-13x13', 2, 7)
run(baseline6x6, 'refnet52x52-6x6', 3, 6)
run(baseline6x6, 'refnet52x52-3x3', 4, 5)
