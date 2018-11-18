import argparse

import numpy as np
from modelzoo.models.gatenet.GateNet import GateNet

from utils.fileaccess.GateGenerator import GateGenerator
from utils.workdir import cd_work
from visuals import demo_generator

if __name__ == '__main__':
    start_idx = 0
    n_repetitions = 1
    cd_work()

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", default=start_idx, type=int)
    parser.add_argument("--n_reps", default=n_repetitions, type=int)

    args = parser.parse_args()

    start_idx = args.start_idx
    n_repetitions = args.n_reps
    anchors = np.array([
        [[81, 82],
         [135, 169],
         [344, 319]],
        [[10, 14],
         [23, 27],
         [37, 58]],
    ])
    architecture = [
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 4, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 8, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 24, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'predict'},
        {'name': 'route', 'index': [-4]},
        {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'upsample', 'size': 2},
        {'name': 'route', 'index': [-1, 8]},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'predict'}
    ]
    model_name = 'yolov3_w0{}x{}'.format(416, 416)
    model = GateNet.create_by_arch(architecture, (208, 208), anchors=anchors,
                                   weight_file='out/thesis/datagen/{0:s}_i{1:02d}/model.h5'.format(model_name, 0),
                                   )

    generator = GateGenerator(directories=['resource/ext/samples/iros2018_course_final_simple_17gates'],
                              batch_size=8, color_format='bgr',
                              shuffle=False, start_idx=400, valid_frac=1.0,
                              label_format='xml',
                              img_format='jpg'
                              )
    demo_generator(model, generator, t_show=0, n_samples=2000, iou_thresh=0.6)
