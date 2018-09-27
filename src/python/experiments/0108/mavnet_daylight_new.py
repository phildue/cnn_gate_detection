import numpy as np

from train import train
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.labels.ObjectLabel import ObjectLabel

img_res = 416, 416

anchors = np.array([[[81, 82],
                     [135, 169],
                     [344, 319]],
                    [[10, 14],
                     [23, 27],
                     [37, 58]]])
architecture = [
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 1024, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 256, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'predict'},
    {'name': 'route', 'index': [-4]},
    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'upsample', 'size': 2},
    {'name': 'route', 'index': [-1, 8]},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'predict'}
]

model_name = 'mavnet_daylight_new{}x{}'.format(img_res[0], img_res[1])
ObjectLabel.classes.append('muro')
train(architecture=architecture,
      work_dir='test/' + model_name,
      img_res=img_res,
      augmenter=RandomEnsemble([
          # (1.0, RandomBrightness(0.5, 1.5)),
          # (0.5, TransformFlip()),
          # (0.1, TransformGray()),
          # (0.25, TransformHistEq()),
          #          (1.0, RandomGrayNoise()),
          #          (0.1, TransformerBlur(iterations=10)),
      ]),
      image_source=['resource/ext/samples/muro'
                    ],
      anchors=anchors,
      epochs=100,
      n_samples=None,
      input_channels=3,
      initial_epoch=0,
      learning_rate=0.001,
      n_polygon=4)
