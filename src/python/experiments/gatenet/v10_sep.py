import numpy as np

from run.training.gate.train import train

grid = [(13, 13)]
img_res = 104, 104
anchors = np.array([[[1, 1],
                     [1 / grid[0][0], 1 / grid[0][1]],  # img_h/img_w
                     [2 / grid[0][0], 2 / grid[0][1]],  # 0.5 img_h/ 0.5 img_w
                     [1 / grid[0][0], 3 / grid[0][0]],  # img_h / 0.33 img_w
                     [1 / grid[0][0], 2 / grid[0][0]]  # img_h / 0.5 img_w
                     ]])

baseline = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'sep_conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
]

for n_layers in range(6):
    architecture = baseline.copy()

    architecture.extend(
        n_layers * [{'name': 'sep_conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1}])

    train(architecture=architecture,
          work_dir='gatenet{}x{}-{}x{}+{}layers+pyramid'.format(img_res[0], img_res[1], grid[0][0],
                                                                grid[0][1], n_layers + 3),
          img_res=img_res,
          anchors=anchors,
          epochs=50,
          n_samples=None)
