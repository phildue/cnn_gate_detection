import numpy as np

from training.gate.train import train
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomColorShift import RandomColorShift
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomGrayNoise import RandomGrayNoise
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransformHistEq import TransformHistEq
from utils.imageprocessing.transform.TransformerBlur import TransformerBlur
from utils.imageprocessing.transform.TransfromGray import TransformGray

grid = [(13, 13)]
img_res = 416, 416
anchors = np.array([[[1.08, 1.19],
                     [3.42, 4.41],
                     [6.63, 11.38],
                     [9.42, 5.11],
                     [16.62, 10.52]]])

architecture = [
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
]

model_name = '1807/narrow{}x{}-{}x{}+{}layers'.format(img_res[0], img_res[1], grid[0][0],
                                                      grid[0][1], 9)

train(architecture=architecture,
      work_dir=model_name,
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
      epochs=100,
      n_samples=None,
      input_channels=3,
      initial_epoch=0)
