import numpy as np

from run.training.cropnet.train import train
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.TransformFlip import TransformFlip

grid = [(13, 13)]
img_res = 52, 52
anchors = [np.array([1,
                     1 / grid[0][0],  # img_h/img_w
                     2 / grid[0][0],  # 0.5 img_h/ 0.5 img_w
                     3 / grid[0][0],  # img_h / 0.33 img_w
                     ])]

baseline = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},

]

for n_layers in range(6):
    architecture = baseline.copy()

    architecture.extend(
        n_layers * [{'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1}])

    train(architecture=architecture,
          work_dir='cropnet{}x{}-{}x{}+{}layers+pyramid'.format(img_res[0], img_res[1], grid[0][0],
                                                                grid[0][1], n_layers + 3),
          img_res=img_res,
          augmenter=RandomEnsemble([(1.0, RandomBrightness(0.5, 2.0)),
                                    (0.5, TransformFlip()),
                                    ]),
          anchor_scale=anchors,
          encoding='anchor',
          epochs=50,
          n_samples=None)
