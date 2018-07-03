# How many filters/layers do we need for a 52x52 crop?

import numpy as np

from run.training.cropnet.train import train
from utils.workdir import cd_work

cd_work()

for img_res in [(52, 52), (104, 104), (26, 26)]:
    for i, grid in enumerate([[(13, 13)], [(6, 6)], [(3, 3)]]):
        anchors = [np.array([1,
                             1 / grid[0][0],  # img_h/img_w
                             2 / grid[0][0],  # 0.5 img_h/ 0.5 img_w
                             4 / grid[0][0],  # img_h / 0.33 img_w
                             5 / grid[0][0]  # img_h / 0.5 img_w
                             ])]
        for width in [64, 32, 16]:

            pool_size = 2
            pooling_layers = np.log(img_res[0] / grid[0][0]) / np.log(pool_size)

            baseline = int(np.floor(pooling_layers)) * [
                {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                {'name': 'max_pool', 'size': (pool_size, pool_size)}]

            if pooling_layers % 1:
                baseline.extend([
                    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)}
                ])

            for n_layers in range(6 - int(len(baseline) / 2), 0, -1):
                architecture = baseline.copy()

                for j in range(n_layers):
                    architecture.append(
                        {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1),
                         'alpha': 0.1})
                architecture.append(
                    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 4, 'strides': (1, 1),
                     'alpha': 0.1})

                train(architecture=architecture,
                      work_dir='cropnet_anch{}x{}-{}x{}+{}layers+{}filters'.format(img_res[0], img_res[1], grid[0][0],
                                                                                   grid[0][1], len(architecture) - int(
                              len(baseline) / 2) + n_layers, width),
                      img_res=img_res,
                      epochs=50,
                      n_samples=None,
                      encoding='anchor',
                      anchor_scale=anchors)
