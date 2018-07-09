import numpy as np

from run.training.gate.train import train
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.TransformFlip import TransformFlip

grid = [(13, 13)]
img_res = 208, 208
anchors = np.array([[[1.08, 1.19],
                     [3.42, 4.41],
                     [6.63, 11.38],
                     [9.42, 5.11],
                     [16.62, 10.52]]])

baseline = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'sep_conv_leaky', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'sep_conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'sep_conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
]

for n_layers in range(6):
    architecture = baseline.copy()

    architecture.extend(
        n_layers * [{'name': 'sep_conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1}])

    train(architecture=architecture,
          work_dir='mobilgatenet{}x{}-{}x{}+{}layers+pyramid'.format(img_res[0], img_res[1], grid[0][0],
                                                                     grid[0][1], n_layers + 3),
          img_res=img_res,
          augmenter=RandomEnsemble([(1.0, RandomBrightness(0.5, 2.0)),
                                    (0.5, TransformFlip()),
                                    ]),
          anchors=anchors,
          epochs=50,
          n_samples=None)
