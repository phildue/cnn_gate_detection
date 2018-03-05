import os

from modelzoo.evaluation.ModelEvaluator import ModelEvaluator
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dir
from utils.workdir import work_dir

work_dir()

name = 'test'

# Image Source
BATCH_SIZE = 4
n_batches = 2
image_source = 'resource/samples/mult_gate_aligned_test/'
color_format = 'bgr'

# Model
conf_thresh = 0
weight_file = 'logs/yolov2_25k/YoloV2.h5'
model = Yolo.yolo_v2(class_names=['gate'], weight_file=weight_file, conf_thresh=conf_thresh, color_format='yuv')

# Evaluator
iou_thresh = 0.4

# Result Paths
result_path = 'logs/yolov2_10k/' + name + '/'
result_file = 'result_' + name
result_img_path = result_path + 'images_' + name + '/'
exp_param_file = 'experiment_parameters_' + name + '.txt'

create_dir([result_path, result_img_path])

generator = GateGenerator(directory=image_source, batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=False, color_format=color_format)

evaluator = ModelEvaluator(model, out_file=result_path + result_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)

exp_params = {'name': name,
              'model': model.net.__class__.__name__,
              'train_samples': '10k',
              'conf_thresh': conf_thresh,
              'iou_thresh': iou_thresh,
              'weight_file': weight_file,
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * BATCH_SIZE}

save_file(exp_params, exp_param_file, result_path)
