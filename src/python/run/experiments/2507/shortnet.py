import numpy as np

from training.gate.train import train
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransformHistEq import TransformHistEq
from utils.imageprocessing.transform.TransfromGray import TransformGray

grid = [(13, 13)]
img_res = 208, 208
anchors = np.array([[[1.08, 1.19],
                     [3.42, 4.41],
                     [6.63, 11.38],
                     [9.42, 5.11],
                     [16.62, 10.52]]])

architecture = [
    # First layer it does not see complex shapes so we apply a few large filters for efficiency
    {'name': 'conv_leaky', 'kernel_size': (5, 5), 'filters': 16, 'strides': (2, 2), 'alpha': 0.1},
    # Second layers we use more but smaller filters to combine shapes in non-linear fashion
    {'name': 'bottleneck_conv', 'kernel_size': (3, 3), 'filters': 24, 'strides': (2, 2), 'alpha': 0.1,
     'compression': 0.5},
    {'name': 'bottleneck_conv', 'kernel_size': (3, 3), 'filters': 30, 'strides': (2, 2), 'alpha': 0.1,
     'compression': 0.5},
    {'name': 'bottleneck_conv', 'kernel_size': (3, 3), 'filters': 40, 'strides': (2, 2), 'alpha': 0.1,
     'compression': 0.5},

    {'name': 'bottleneck_conv', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
     'compression': 0.5},
    # Final layers, the shapes should be "exhausted" now its about combining spatial information
    # That is why we increase kernel size to collect it
    {'name': 'bottleneck_conv', 'kernel_size': (7, 7), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
     'compression': 0.5},

    # {'name': 'bottleneck_conv', 'kernel_size': (9, 9), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1,
    #  'compression': 0.5},

    # {'name': 'bottleneck_conv', 'kernel_size': (9, 9), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1,
    # 'compression': 0.5},
    # receptive field is problematic as the final layer does not see the object
    # if this network works well we should use dilated convolutions in the later layers

]

model_name = 'shortnet161{}x{}-{}x{}+{}layers'.format(img_res[0], img_res[1], grid[0][0],
                                                      grid[0][1], 9)

train(architecture=architecture,
      work_dir='2507/' + model_name,
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
