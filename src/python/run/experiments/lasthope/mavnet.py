import numpy as np

from training.gate.train import train
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransfromGray import TransformGray

img_res = 208, 208
anchors = np.array([
    [[30, 30],
     [48, 16],
     [60, 60]],
    [[90, 39],
     [200, 14],
     [120, 120]]
])

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
    {'name': 'predict'},

    {'name': 'bottleneck_conv', 'kernel_size': (9, 9), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1,
     'compression': 0.5},

    {'name': 'bottleneck_conv', 'kernel_size': (9, 9), 'filters': 24, 'strides': (1, 1), 'alpha': 0.1,
     'compression': 0.5},
    {'name': 'predict'},

    # receptive field is problematic as the final layer does not see the object
    # if this network works well we should use dilated convolutions in the later layers

]

model_name = 'mavnet{}x{}-jevois'.format(img_res[0], img_res[1])

train(architecture=architecture,
      work_dir='last_hope/' + model_name,
      img_res=img_res,
      augmenter=RandomEnsemble([
          (1.0, RandomBrightness(0.5, 1.5)),
          (0.5, TransformFlip()),
          (0.1, TransformGray()),
          # (0.25, TransformHistEq()),
          #          (1.0, RandomGrayNoise()),
          #          (0.1, TransformerBlur(iterations=10)),
      ]),
      anchors=anchors,
      epochs=100,
      n_samples=None,
      input_channels=3,
      initial_epoch=0,
      image_source=['resource/ext/samples/daylight_course1',
                    'resource/ext/samples/daylight_course5',
                    'resource/ext/samples/daylight_course3',
                    'resource/ext/samples/iros2018_course1',
                    'resource/ext/samples/iros2018_course5',
                    'resource/ext/samples/iros2018_flights',
                    'resource/ext/samples/basement20k',
                    'resource/ext/samples/basement15k',
                    'resource/ext/samples/basement_course3',
                    'resource/ext/samples/basement_course1',
                    'resource/ext/samples/iros2018_course3_test'],
      batch_size=16,
      color_format='bgr')
