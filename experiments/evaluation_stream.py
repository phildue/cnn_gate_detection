import os


from workdir import work_dir

work_dir()

from models.Yolo.Yolo import Yolo
from fileaccess.GateGenerator import GateGenerator
from evaluation.MetricOneGate import MetricOneGate
from evaluation.BasicEvaluator import BasicEvaluator
from fileaccess.utils import save

# Image Source
BATCH_SIZE = 100
n_batches = 6
image_source = 'resource/samples/stream_valid2/'
color_format = 'bgr'

# Result Paths
result_path = 'logs/yolo-noaug/'
result_file = 'test-stream2-iou0.4.pkl'
result_img_path = result_path + '/stream/2/'

# Model
conf_thresh = 0.3
weight_file = 'logs/yolo-nocolor/yolo-gate-adam.h5'
model = Yolo(class_names=['gate'], weight_file=weight_file, conf_thresh=conf_thresh)

# Evaluator
iou_thresh = 0.4

if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)

generator = GateGenerator(directory=image_source, batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=False, color_format=color_format)

evaluator = BasicEvaluator(model,
                           metrics=[MetricOneGate(iou_thresh=iou_thresh, show_=True, store_path=result_img_path)],
                           out_file=result_path + result_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)

exp_params = {'model': model.__class__.__name__,
              'conf_thresh': conf_thresh,
              'iou_thresh': iou_thresh,
              'weight_file': weight_file,
              'image_source': image_source,
              'color_format': color_format,
              'n_samples': n_batches * BATCH_SIZE}

save(exp_params, 'experiment_parameters.pkl', result_path)
