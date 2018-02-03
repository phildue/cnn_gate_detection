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
from evaluation.MetricOneGate import MetricOneGate
from evaluation.BasicEvaluator import BasicEvaluator

BATCH_SIZE = 100
n_batches = 6
result_path = 'logs/yolo-nocolor/'
stream_path = '/stream/2/'
if not os.path.exists(result_path + stream_path):
    os.makedirs(result_path + stream_path)

experiment_file = 'test-stream2-iou0.4.pkl'
generator = GateGenerator(directory='resource/samples/stream_valid2/', batch_size=BATCH_SIZE, img_format='jpg',
                          shuffle=False, color_format='bgr')

yolo = Yolo(class_names=['gate'], weight_file='logs/yolo-nocolor/yolo-gate-adam.h5', conf_thresh=0.3)
#
# EvaluatorPrecisionRecall(yolo, iou_thresh=0.6).evaluate_generator(set, n_batches=n_batches,
#                                                                   out_file=result_path + experiment_file)

evaluator = BasicEvaluator(yolo,
                           metrics=[MetricOneGate(iou_thresh=0.4, show_=True, store_path=result_path + stream_path)],
                           out_file=result_path + experiment_file)

evaluator.evaluate_generator(generator, n_batches=n_batches)
