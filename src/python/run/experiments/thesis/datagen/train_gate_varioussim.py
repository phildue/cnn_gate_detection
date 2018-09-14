import argparse

import numpy as np

from training.gate.train import train

if __name__ == '__main__':
    start_idx = 0
    n_repetitions = 5
    img_res = 416, 416

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", default=start_idx, type=int)
    parser.add_argument("--n_reps", default=n_repetitions, type=int)

    args = parser.parse_args()

    start_idx = args.start_idx
    n_repetitions = args.n_reps
    anchors = np.array([
        [[10, 14],
         [23, 27],
         [37, 58]],
        [[81, 82],
         [135, 169],
         [344, 319]],
    ])
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

    model_name = 'yolov3_gate_varioussim{}x{}'.format(img_res[0], img_res[1])

    augmenter = None

    image_source = ['resource/ext/samples/various_environments20k']

    for i in range(start_idx, n_repetitions):
        train(architecture=architecture,
              work_dir='thesis/datagen/' + model_name + '_i{0:02d}'.format(i),
              img_res=img_res,
              augmenter=augmenter,
              image_source=image_source,
              anchors=anchors,
              epochs=100,
              batch_size=16,
              n_samples=None,
              min_obj_size=0.1,
              max_obj_size=1.2,
              min_aspect_ratio=0.3,
              max_aspect_ratio=3.0,
              initial_epoch=0,
              color_format='bgr')
