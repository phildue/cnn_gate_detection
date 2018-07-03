import numpy as np

from run.training.gate.train import train

grid = [(13, 13)]
img_res = 52, 52
anchors = np.array([[[1, 1],
                     [1 / grid[0][0], 1 / grid[0][1]],  # img_h/img_w
                     [2 / grid[0][0], 2 / grid[0][1]],  # 0.5 img_h/ 0.5 img_w
                     [1 / grid[0][0], 3 / grid[0][0]],  # img_h / 0.33 img_w
                     [1 / grid[0][0], 2 / grid[0][0]]  # img_h / 0.5 img_w
                     ]])

architecture = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
]


train(architecture=architecture,
      work_dir='gatenet-test'.format(img_res[0], img_res[1], grid[0][0],
                                                            grid[0][1], 9 + 3),
      img_res=img_res,
      anchors=anchors,
      epochs=50,
      n_samples=None)
