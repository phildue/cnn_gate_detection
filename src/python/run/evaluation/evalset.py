import os

import numpy as np

from modelzoo.evaluation.ModelEvaluator import ModelEvaluator
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import cd_work

cd_work()

name = '2803'

# Image Source
batch_size = 73
n_batches = 1
image_source = ['resource/samples/video/eth']
color_format = 'bgr'

# Model
conf_thresh = 0
weight_file = 'logs/v2_bebop_distort/YoloV2.h5'

anchors = np.array([[0.13809687, 0.27954467],
                    [0.17897748, 0.56287585],
                    [0.36642611, 0.39084195],
                    [0.60043528, 0.67687858]])

model = Yolo.yolo_v2(norm=(160, 320), grid=(5, 10), class_names=['gate'], batch_size=4,
                       color_format='bgr', anchors=anchors,conf_thresh=conf_thresh,weight_file=weight_file)
# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'logs/v2_bebop_distort/' + name + '/'
result_file = 'result_' + name
result_img_path = result_path + 'images_' + name + '/'
exp_param_file = 'experiment_parameters_' + name + '.txt'

create_dirs([result_path, result_img_path])

generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg',
                          shuffle=False, color_format=color_format, label_format='pkl',start_idx=73)

evaluator = ModelEvaluator(model, out_file=result_path + result_file,)

evaluator.evaluate_generator(generator, n_batches=n_batches)

exp_params = {'name': name,
              'model': model.net.__class__.__name__,
              'train_samples': '10k',
              'conf_thresh': conf_thresh,
              'iou_thresh': iou_thresh,
              'weight_file': weight_file,
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * batch_size}

save_file(exp_params, exp_param_file, result_path)
