import os

from evaluation.SpeedEvaluator import SpeedEvaluator
from models.yolo.TinyYolo import TinyYolo

from workdir import work_dir

work_dir()

from models.yolo.Yolo import Yolo
from fileaccess.GateGenerator import GateGenerator
from evaluation.MetricOneGate import MetricOneGate
from evaluation.BasicDetectionEvaluator import BasicDetectionEvaluator
from fileaccess.utils import save

name = 'speed'

# Image Source
BATCH_SIZE = 10
n_batches = 6
image_source = 'resource/samples/stream_valid2/'
color_format = 'bgr'

# Model
conf_thresh = 0.3
weight_file = 'logs/tinyyolo-noaug/yolo-gate-adam.h5'
model = TinyYolo(class_names=['gate'], weight_file=weight_file, conf_thresh=conf_thresh)

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'logs/yolo-noaug/' + name + '/'
result_file = 'result.pkl'
result_img_path = result_path + 'images/'
exp_param_file = 'experiment_parameters.txt'


if not os.path.exists(result_path):
    os.makedirs(result_path)

if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)

generator = GateGenerator(directory=image_source, batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=False, color_format=color_format)

evaluator = SpeedEvaluator(model,
                           out_file=result_path+result_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)

exp_params = {'name': name,
              'model': model.model.__class__.__name__,
              'evaluator': evaluator.__class__.__name__,
              'conf_thresh': conf_thresh,
              'iou_thresh': iou_thresh,
              'weight_file': weight_file,
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * BATCH_SIZE}

save(exp_params, exp_param_file, result_path)
