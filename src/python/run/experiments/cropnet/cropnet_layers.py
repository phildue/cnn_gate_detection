# How many filters/layers do we need for a 52x52 crop?

import numpy as np

from run.training.cropnet.train import train
from utils.workdir import cd_work

cd_work()
for img_res in [(416, 416), (104, 104)]:
    for i, grid in enumerate([[(3, 3)], [(6, 6)], [(13, 13)]]):
        for width in [64, 32, 16]:
            pool_size = 8
            pooling_layers = np.log(img_res[0] / grid[0][0]) / np.log(pool_size)

            baseline = int(np.floor(pooling_layers)) * [
                {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                {'name': 'max_pool', 'size': (pool_size, pool_size)}]

            if pooling_layers % 1:
                baseline.extend([
                    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)}
                ])

            for n_layers in range(8 - int(len(baseline) / 2)):
                architecture = baseline.copy()

                for j in range(n_layers):
                    architecture.append(
                        {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': width, 'strides': (1, 1),
                         'alpha': 0.1})

                train(architecture=architecture,
                      work_dir='cropnet{}x{}->{}x{}+{}layers+{}filters'.format(img_res[0], img_res[1], grid[0][0],
                                                                               grid[0][1], len(architecture) - int(
                              len(baseline) / 2) + n_layers, width),
                      img_res=img_res,
                      epochs=50,
                      n_samples=None)
