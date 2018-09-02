import numpy as np

from training.yolo.train import train

n_repetitions = 10
img_res = 416, 416
# anchors = np.array([[[2.53, 2.54],
#                      [4.21, 5.28],
#                      [10.75, 9.96]],
#                     [[0.62, 0.86],
#                      [1.4, 1.9],
#                      [2.31, 3.625],
#                      ]])
anchors = np.array([[[1.08, 1.19],
                     [3.42, 4.41],
                     [6.63, 11.38]],
                    [[9.42, 5.11],
                     [16.62, 10.52]]])
# 10,14, 23,27, 37,58, 81,82, 135,169, 344,319
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

model_name = 'yolov3_person{}x{}'.format(img_res[0], img_res[1])

augmenter = None

image_source = ['resource/ext/samples/muro']

for i in range(n_repetitions):
    train(architecture=architecture,
          work_dir='thesis/datagen' + model_name + '_i{0:02d}'.format(i),
          img_res=img_res,
          augmenter=augmenter,
          image_source=image_source,
          anchors=anchors,
          epochs=100,
          n_samples=None,
          class_names='muro',
          initial_epoch=0)
