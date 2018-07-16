from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
import numpy as np




cd_work()

"""
Refnet

"""
# for f in [32]:
#     precisions = []
#     recalls = []
#     for iou in [0.4, 0.6, 0.8]:
#         content = load_result('refnet', (52, 52), (3, 3), 4, 16, False, 'result_' + str(iou) + '.pkl')
#         results = content['results']['MetricDetection']
#         results = [ResultByConfidence(d) for d in results]
#
#         results_sum = results[0]
#         for r in results[1:]:
#             results_sum += r
#         precision = []
#         recall = []
#         for c in results_sum.confidences:
#             print(results_sum.results[c])
#             precision.append(results_sum.results[c].precision)
#             recall.append(results_sum.results[c].recall)
#         precisions.append(precision)
#         recalls.append(recall)
#     BaseMultiPlot(
#         x_data=recalls,
#         y_data=precisions,
#         x_res=None
#     ).show()

# content = load_file('out/gatev49/results/set_1-040.pkl')
# results = content['results']['MetricDetection']
# results = [ResultByConfidence(d) for d in results]
#
# results_sum = results[0]
# for r in results[1:]:
#     results_sum += r
# precision = []
# recall = []
# for c in results_sum.confidences:
#     print(results_sum.results[c])
#     precision.append(results_sum.results[c].precision)
#     recall.append(results_sum.results[c].recall)
# BasePlot(
#     x_data=recall,
#     y_data=precision,
# ).show()

pr_plot(['out/gatev49/results/set_1-040.pkl'],
        ['5'], 'CropNet').show(False)

detection_plot('out/gatev49/results/set_1-040.pkl','CropNet').show()
