import os

from frontend.evaluation.ModelEvaluator import ModelEvaluator

from frontend.evaluation.ConfidenceEvaluator import ConfidenceEvaluator
from frontend.evaluation.MetricDetection import MetricDetection
from frontend.models.yolo.Yolo import Yolo
from workdir import work_dir

work_dir()

from fileaccess.GateGenerator import GateGenerator
from fileaccess.utils import save_file

name = 'test'

# Source
in_file = 'logs/yolov2_10k/test/result_test.pkl'
color_format = 'bgr'

# Model
conf_thresh = 0
weight_file = 'logs/yolov2_25k/YoloV2.h5'
model = Yolo.yolo_v2(class_names=['gate'], weight_file=weight_file, conf_thresh=conf_thresh, color_format='yuv')

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'logs/yolov2_10k/' + name + '/'
result_file = 'metric_result_' + name
result_img_path = result_path + 'images_' + name + '/'
exp_param_file = 'experiment_parameters_' + name + '.txt'

if not os.path.exists(result_path):
    os.makedirs(result_path)

if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)

evaluator = ConfidenceEvaluator(model, metrics=[MetricDetection(show_=True)], out_file=result_path + result_file,
                                color_format=color_format)

evaluator.evaluate(in_file)

exp_params = {'name': name,
              'model': model.net.__class__.__name__,
              'train_samples': '10k',
              'conf_thresh': conf_thresh,
              'iou_thresh': iou_thresh,
              'weight_file': weight_file,
              'in_source': in_file,
              'color_format': color_format}

save_file(exp_params, exp_param_file, result_path)