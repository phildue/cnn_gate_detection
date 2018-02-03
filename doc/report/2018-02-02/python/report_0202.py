import os
import sys

import numpy as np


PROJECT_ROOT = '/home/phil/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from fileaccess.utils import load
from evaluation.EvaluatorPrecisionRecall import EvaluatorPrecisionRecall
from evaluation.ResultsByConfidence import ResultByConfidence
from backend.plots.BasePlot import BasePlot
from backend.plots.PlotPrecisionRecallMulti import PlotPrecisionRecallMulti
from evaluation.DetectionResult import DetectionResult
from backend.videomaker import make_video
from evaluation.StreamAnalyzer import StreamAnalyzer

result_path = 'dvlab/src/python/doc/report_0202/'
doc_path = '../../doc/report/0202/'

# ---------------------------------------------------------------------------- Test 1000 IoU 0.4
# experiment_file = 'test1000-iou0.4.pkl'
# experiments = load(result_path + experiment_file)
#
# detection_result = experiments['MetricDetection']
# detection_result = [ResultByConfidence(d) for d in detection_result]
# total = detection_result[0]
# for result in detection_result[1:]:
#     total = total + result
#
# precision_04, recall_04 = EvaluatorPrecisionRecall.interp(total)
#
# localization_error = experiments['MetricLocalization']
#
# result_mat = np.vstack([r[0.1] for r in localization_error if r[0.1] is not None])
#
# print("Average Localization Error")
# for i, d in enumerate(['CX', 'CY', 'W', 'H']):
#     print(d + ': {0:02f} +/- {1:02f}'.format(np.mean(result_mat[:, i]), np.std(result_mat[:, i])))
#
# print("Average Mean Localization Error")
# print(' {0:02f} +/- {1:02f}'.format(np.mean(result_mat), np.std(result_mat)))

# ---------------------------------------------------------------------------- Test 1000 IoU 0.6
# experiment_file = 'test1000-iou0.6.pkl'
# experiments = load(result_path + experiment_file)
#
# detection_result = experiments['MetricDetection']
# detection_result = [ResultByConfidence(d) for d in detection_result]
# total = detection_result[0]
# for result in detection_result[1:]:
#     total = total + result
#
# precision_06, recall_06 = EvaluatorPrecisionRecall.interp(total)
#
# experiment_file = 'test1000-iou0.4.pkl'
# experiments = load('logs/tiny-yolo-gate-mult-2/' + experiment_file)
#
# detection_result = experiments['MetricDetection']
# detection_result = [ResultByConfidence(d[0]) for d in detection_result]
# total = detection_result[0]
# for result in detection_result[1:]:
#     total = total + result
#
# tiny_precision_04, tiny_recall_04 = EvaluatorPrecisionRecall.interp(total)
#
# plot_pr = PlotPrecisionRecallMulti(recall=[recall_04, recall_06, tiny_recall_04],
#                                    precision=[precision_04, precision_06, tiny_precision_04],
#                                    legend=['IoU=0.4', 'IoU=0.6', 'Tiny IoU=0.4'], size=(5, 4), font_size=8,
#                                    line_style=['r--', 'g--', 'b--'])
#
# plot_pr.save(doc_path + '/precision-recall.png')

# --------------------------------------------------------------------------------------------------Stream 1
print(">>>>>>>>>>>>>>>>>>>>>>>>>Stream1")

# make_video(result_path + '/stream/1/', output=doc_path + 'stream1.avi')
analyzer = StreamAnalyzer(result_path+'test-stream1-iou0.4.pkl')
analyzer.loc_error_plot().show()
analyzer.detection_eval()

# --------------------------------------------------------------------------------------------------Stream 2
print(">>>>>>>>>>>>>>>>>>>>>>>>>Stream2")

#make_video(result_path + '/stream/2/', output=doc_path + 'stream2.avi')
analyzer = StreamAnalyzer(result_path+'test-stream2-iou0.4.pkl')
analyzer.loc_error_plot().show()
analyzer.detection_eval()