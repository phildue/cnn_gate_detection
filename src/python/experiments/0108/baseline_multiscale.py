from train import train
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.imageprocessing.transform.TransformHistEq import TransformHistEq
from utils.imageprocessing.transform.TransfromGray import TransformGray

grid = [(13, 13),
        (7, 7),
        (3, 3),
        (1, 1)]
img_res = 416, 416

anchors = [
    [[1, 1], [1.5, 0.5]],
    [[1, 1], [1.5, 0.5]],
    [[1, 1], [1.5, 0.5]],
    # [[1, 1], [1.5, 0.5]],
    [[1, 1], [1.5, 0.5], [2.5, 0.25]]
]

architecture = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'predict'},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'predict'},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'predict'},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'predict'},

]

model_name = 'baseline_multiscale{}x{}'.format(img_res[0], img_res[1])

train(architecture=architecture,
      work_dir='0108/' + model_name,
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
