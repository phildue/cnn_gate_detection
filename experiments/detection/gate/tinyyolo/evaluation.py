import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
PROJECT_ROOT = '/home/phil/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from fileaccess.GateGenerator import GateGenerator
from evaluation.MetricDetection import MetricDetection
from evaluation.ConfidenceEvaluator import ConfidenceEvaluator
from models.Yolo.TinyYolo import TinyYolo
from evaluation.MetricLocalization import MetricLocalization

BATCH_SIZE = 100
n_batches = 10
result_path = 'logs/tiny-yolo-gate-mult-2/'

experiment_file = 'test1000-iou0.4.pkl'
generator = GateGenerator(directory='resource/samples/mult_gate_valid/', batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=True, color_format='bgr')

yolo = TinyYolo(class_names=['gate'], weight_file='logs/tiny-yolo-gate-mult-2/yolo-gate-adam.h5')

evaluator = ConfidenceEvaluator(yolo, metrics=[MetricDetection(iou_thresh=0.4), MetricLocalization(iou_thresh=0.4)],
                                out_file=result_path + experiment_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)
