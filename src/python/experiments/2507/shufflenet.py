import numpy as np
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness

from modelzoo.train import train
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransformHistEq import TransformHistEq
from utils.imageprocessing.transform.TransfromGray import TransformGray

grid = [(13, 13)]
img_res = 416, 416
anchors = np.array([[[1.08, 1.19],
                     [3.42, 4.41],
                     [6.63, 11.38],
                     [9.42, 5.11],
                     [16.62, 10.52]]])

architecture = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'bottleneck_dconv', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1, 'expansion': 6},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'bottleneck_dconv', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1, 'expansion': 6},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'bottleneck_dconv', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1, 'expansion': 6},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'bottleneck_dconv', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1, 'expansion': 6},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'bottleneck_dconv_residual', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
     'expansion': 6},
    {'name': 'bottleneck_dconv_residual', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
     'expansion': 6},
    {'name': 'bottleneck_dconv_residual', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
     'expansion': 6},

]

train(architecture=architecture,
      work_dir='1807/mobilegatenetV2{}x{}-{}x{}+{}layers'.format(img_res[0], img_res[1], grid[0][0],
                                                                 grid[0][1], 9),
      img_res=img_res,
      augmenter=RandomEnsemble([
          (1.0, RandomBrightness(0.5, 1.5)),
          (0.5, TransformFlip()),
          (0.1, TransformGray()),
          (0.25, TransformHistEq()),
          #          (1.0, RandomGrayNoise()),
          #          (0.1, TransformerBlur(iterations=10)),
      ]),
      anchors=anchors,
      epochs=50,
      n_samples=None)
