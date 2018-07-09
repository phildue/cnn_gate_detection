import numpy as np

from run.training.gate.train import train

grid = [(13, 13)]
img_res = 52, 52
anchors = np.array([[[1.08, 1.19],
                     [3.42, 4.41],
                     [6.63, 11.38],
                     [9.42, 5.11],
                     [16.62, 10.52]]])
architecture = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
]


train(architecture=architecture,
      work_dir='gatenet-test',
      img_res=img_res,
      anchors=anchors,
      epochs=50,
      n_samples=None)
