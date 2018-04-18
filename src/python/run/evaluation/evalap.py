import os

import numpy as np

from modelzoo.evaluation.ConfidenceEvaluator import ConfidenceEvaluator
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.utils import create_dirs, save_file, load_file
from utils.workdir import cd_work

cd_work()

name = '1804'
op_dir = 'logs/gatev0_industrial/'
# Source
in_file = op_dir + name + '/result_1804.pkl'
color_format = 'bgr'

# Model
conf_thresh = 0
weight_file = op_dir + '/GateNetV0.h5'

model = GateNet.v0(weight_file=weight_file, conf_thresh=0.001)

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = op_dir + name + '/'
result_file = 'metric_result_' + name
result_img_path = result_path + 'images_' + name + '/'
exp_param_file = 'experiment_parameters_' + name + '.txt'

create_dirs([result_path, result_img_path])

evaluator = ConfidenceEvaluator(model, metrics=[MetricDetection(show_=False)], out_file=result_path + result_file,
                                color_format=color_format)
file_content = load_file(in_file)
labels_true = file_content['labels_true']
image_files = file_content['image_files']
labels_pred = file_content['labels_pred']

evaluator.evaluate(labels_true, labels_pred, image_files)

exp_params = {'name': name,
              'model': model.net.__class__.__name__,
              'train_samples': '20k',
              'conf_thresh': conf_thresh,
              'iou_thresh': iou_thresh,
              'weight_file': weight_file,
              'in_source': in_file,
              'color_format': color_format}

save_file(exp_params, exp_param_file, result_path)
