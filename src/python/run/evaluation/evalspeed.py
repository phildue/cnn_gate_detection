import os

from modelzoo.evaluation.SpeedEvaluator import SpeedEvaluator
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import cd_work

cd_work()


name = 'speed'

# Image Source
BATCH_SIZE = 10
n_batches = 6
image_source = 'resource/samples/stream_valid2/'
color_format = 'bgr'

# Model
conf_thresh = 0.3
weight_file = 'logs/tinyyolo_10k/TinyYolo.h5'
model = Yolo.tiny_yolo(class_names=['gate'], weight_file=weight_file, conf_thresh=conf_thresh)

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'logs/tinyyolo_10k/' + name + '/'
result_file = 'result.pkl'
result_img_path = result_path + 'images/'
exp_param_file = 'experiment_parameters.txt'

create_dirs([result_path, result_img_path])

generator = GateGenerator(directories=image_source, batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=False, color_format=color_format)

evaluator = SpeedEvaluator(model,
                           out_file=result_path+result_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)

exp_params = {'name': name,
              'model': model.net.__class__.__name__,
              'evaluator': evaluator.__class__.__name__,
              'conf_thresh': conf_thresh,
              'iou_thresh': iou_thresh,
              'weight_file': weight_file,
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * BATCH_SIZE}

save_file(exp_params, exp_param_file, result_path)
