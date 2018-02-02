import os
import sys
from os.path import expanduser

import numpy as np


PROJECT_ROOT = expanduser('~') + '/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from fileaccess.utils import load
from evaluation.EvaluatorPrecisionRecall import EvaluatorPrecisionRecall
from backend.plots.PlotPrecisionRecall import PlotPrecisionRecall
from visualization.PositionPlotCreator import PositionPlotCreator
from evaluation.ResultsByConfidence import ResultByConfidence
from evaluation.StreamAnalyzer import StreamAnalyzer

result_path = 'logs/yolo-gate-mult-05/'
# experiment_file = 'test1000-iou0.6.pkl'
# experiments = load(result_path + experiment_file)
# # experiments.extend(load(result_path + 'experiment_results_5000.pkl'))
# # experiments.extend(load(result_path + 'experiment_results.pkl'))
#
# # group_and_plot(experiments, output_path='../../doc/poster/fig/', n_bins=60, block=False, fig_size=(11, 5), fontsize=24)#
#
# detection_result = experiments['MetricDetection']
# detection_result = [ResultByConfidence(d) for d in detection_result]
# total = detection_result[0]
# for result in detection_result[1:]:
#     total = total + result
#
# localization_error = experiments['MetricLocalization']
#
# result_mat = np.vstack([r[0.1] for r in localization_error if r[0.1] is not None])
#
# print(np.mean(result_mat, axis=0))
# print(np.std(result_mat, axis=0))
#
# precision, recall = EvaluatorPrecisionRecall.interp(total)
#PlotPrecisionRecall(precision, recall).show(block=True)


analyzer = StreamAnalyzer(result_path+'test-stream2-iou0.4.pkl')
analyzer.loc_error_plot().show()
analyzer.detection_eval()
# results_label = [rl for rl in experiments if rl[1] is not None]
# PositionPlotCreator(results_label).create('eucl', 'Euclidian Distance', '.').show(block=False)
# PositionPlotCreator(results_label).create_bin('eucl', 'Euclidian Distance', '.', bin_size=50).show(block=True)
