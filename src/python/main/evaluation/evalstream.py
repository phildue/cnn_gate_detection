import os

from modelzoo.evaluation.DetectionEvaluator import DetectionEvaluator
from modelzoo.evaluation.MetricOneGate import MetricOneGate
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.workdir import work_dir

work_dir()


name = 'stream3'

# Image Source
BATCH_SIZE = 4
n_batches = 150
image_source = 'resource/samples/stream_valid3/'
color_format = 'bgr'

# Model
conf_thresh = 0.1
weight_file = 'logs/yolov2_10k/YoloV2.h5'
model = Yolo.yolo_v2(class_names=['gate'], weight_file=weight_file, conf_thresh=conf_thresh)

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'logs/yolov2_10k/' + name + '/'
result_file = 'result.pkl'
result_img_path = result_path + 'images/'
exp_param_file = 'experiment_parameters.txt'

create_dirs([result_path, result_img_path])

generator = GateGenerator(directory=image_source, batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=False, color_format=color_format)

evaluator = DetectionEvaluator(model,
                               metrics=[MetricOneGate(iou_thresh=iou_thresh, show_=True, store_path=result_img_path)],
                               out_file=result_path + result_file)

evaluator.evaluate(generator, n_batches=n_batches)

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
