# How many filters/layers do we need for a 52x52 crop?

import numpy as np

from run.training.refine.train import train
from utils.workdir import cd_work

cd_work()
img_res = 52, 52
grids = [[(3, 3)], [(6, 6)], [(13, 13)]]

for i, grid in enumerate(grids):
    anchors = np.array([[[1, 1],
                         [1 / grid[0][0], 1 / grid[0][1]],  # img_h/img_w
                         [2 / grid[0][0], 2 / grid[0][1]],  # 0.5 img_h/ 0.5 img_w
                         [1 / grid[0][0], 3 / grid[0][0]],  # img_h / 0.33 img_w
                         [1 / grid[0][0], 2 / grid[0][0]]  # img_h / 0.5 img_w
                         ]])
    for width in [64, 32, 16]:

        baseline = int(np.log2(img_res[0] / grid[0][0])) * [
            {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)}]

        for n_layers in range(6 - int(len(baseline) / 2)):
            architecture = baseline.copy()

            for j in range(n_layers):
                architecture.append(
                    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1})

            train(architecture=architecture,
                  work_dir='refnet{}x{}->{}x{}+{}layers+{}filters'.format(img_res[0], img_res[1], grid[0][0],
                                                                          grid[0][1], len(architecture) - int(
                          len(baseline) / 2) + n_layers, width),
                  img_res=img_res,
                  anchors=anchors,
                  epochs=50,
                  n_samples=None)
