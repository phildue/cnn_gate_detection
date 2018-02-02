import os
import sys
from os.path import expanduser

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
PROJECT_ROOT = expanduser('~') + '/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from models.Yolo.Yolo import Yolo
from fileaccess.GateGenerator import GateGenerator
from evaluation.MetricDetection import MetricDetection
from evaluation.MetricLocalization import MetricLocalization
from evaluation.ConfidenceEvaluator import ConfidenceEvaluator

BATCH_SIZE = 100
n_batches = 10
result_path = 'logs/yolo-gate-mult/'

experiment_file = 'test1000-iou0.6.pkl'
generator = GateGenerator(directory='resource/samples/mult_gate_valid/', batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=True, color_format='bgr')

yolo = Yolo(class_names=['gate'], weight_file='logs/yolo-gate-mult/yolo-gate-adam.h5')
#
# EvaluatorPrecisionRecall(yolo, iou_thresh=0.6).evaluate_generator(set, n_batches=n_batches,
#                                                                   out_file=result_path + experiment_file)

evaluator = ConfidenceEvaluator(yolo, metrics=[MetricDetection(iou_thresh=0.6)],
                                out_file=result_path + experiment_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)

