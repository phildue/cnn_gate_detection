import numpy as np

from train import train
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomHSV import RandomHSV
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransfromGray import TransformGray

img_res = 416, 416
anchors = np.array([
    [[30, 30],
     [48, 16],
     [60, 60]],
    [[90, 39],
     [200, 14],
     [120, 120]]
])

architecture = [
    {'name': 'conv_leaky', 'kernel_size': (5, 5), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 24, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 40, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 48, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (7, 7), 'filters': 56, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'predict'},
    {'name': 'conv_leaky', 'kernel_size': (9, 9), 'filters': 56, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (9, 9), 'filters': 28, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'predict'},


]

model_name = 'arch'.format(img_res[0], img_res[1])

train(architecture=architecture,
      work_dir='thesis/architecture/' + model_name,
      img_res=img_res,
      augmenter=RandomEnsemble([
          (1.0, RandomHSV((0.9, 1.1), (0.5, 1.5), (0.5, 1.5))),
          (0.5, TransformFlip()),
          # (0.25, TransformHistEq()),
          #          (1.0, RandomGrayNoise()),
          #          (0.1, TransformerBlur(iterations=10)),
      ]),
      weight_file='out/last_hope/' + model_name + '/model.h5',
      anchors=anchors,
      min_aspect_ratio=.3,
      max_aspect_ratio=4.0,
      min_obj_size=0.01,
      max_obj_size=2.0,
      epochs=100,
      n_samples=None,
      input_channels=3,
      initial_epoch=8,
      image_source=['resource/ext/samples/daylight_course1',
                    'resource/ext/samples/daylight_course5',
                    'resource/ext/samples/daylight_course3',
                    'resource/ext/samples/iros2018_course1',
                    'resource/ext/samples/iros2018_course5',
                    'resource/ext/samples/iros2018_flights',
                    'resource/ext/samples/real_and_sim',
                    'resource/ext/samples/basement20k',
                    'resource/ext/samples/basement_course3',
                    'resource/ext/samples/basement_course1',
                    'resource/ext/samples/iros2018_course3_test'],
      batch_size=16,
      color_format='bgr')
