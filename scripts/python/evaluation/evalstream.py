import os

from frontend.models.yolo.Yolo import Yolo
from workdir import work_dir

work_dir()

from fileaccess.GateGenerator import GateGenerator
from frontend.evaluation.MetricOneGate import MetricOneGate
from frontend.evaluation.BasicDetectionEvaluator import BasicDetectionEvaluator
from fileaccess.utils import save

name = 'stream3'

# Image Source
BATCH_SIZE = 50
n_batches = 12
image_source = 'resource/samples/stream_valid3/'
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


if not os.path.exists(result_path):
    os.makedirs(result_path)

if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)

generator = GateGenerator(directory=image_source, batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=False, color_format=color_format)

evaluator = BasicDetectionEvaluator(model,
                                    metrics=[MetricOneGate(iou_thresh=iou_thresh, show_=True, store_path=result_img_path)],
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
