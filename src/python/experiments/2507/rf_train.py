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
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    #  {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
]
for i in range(1, 10):
    model_name = 'rf{}'.format(127 + i * 32)

    train(architecture=architecture + {'name': 'conv_leaky', 'kernel_size': (i, i), 'filters': 64, 'strides': (1, 1),
                                       'alpha': 0.1},
          work_dir='2507/receptive_field/' + model_name,
          img_res=img_res,
          augmenter=RandomEnsemble([
              (1.0, RandomBrightness(0.5, 1.5)),
              (0.5, TransformFlip()),
              (0.1, TransformGray()),
              (0.25, TransformHistEq()),
          ]),
          anchors=anchors,
          epochs=50,
          n_samples=None,
          input_channels=3,
          initial_epoch=0)
