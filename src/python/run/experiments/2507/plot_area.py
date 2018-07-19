import os

from modelzoo.backend.visuals.plots.BaseBarPlot import BaseBarPlot
from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
import numpy as np

cd_work()

legends = []
total_precisions = []
linestyle = ['-.', '-*', '-x', '-o', '--']
#
# models = [
#     'baseline104x104-13x13+9layers',
#     'baseline416x416-13x13+9layers',
#     'bottleneck416x416-13x13+9layers',
#     'bottleneck_narrow416x416-13x13+9layers',
#     'narrow416x416-13x13+9layers',
#     'narrow_strides416x416-13x13+9layers',
#     'narrow_strides_late_bottleneck416x416-13x13+9layers',
#     'strides416x416-13x13+9layers'
# ]

models = [
    'baseline416x416-13x13+9layers',
    'baseline104x104-13x13+9layers',
    'narrow416x416-13x13+9layers']

names = models
areas = [0.01, 0.05, 0.1, 0.15, 0.25]
for model in models:
    total_average_precision = []
    for iou_thresh in [0.6]:
        for min_box_area in areas:
            results = load_file(
                'out/2507/areatest/' + model + '/iou{}-area{}_result_metric.pkl'.format(iou_thresh,
                                                                                        min_box_area))
            detections = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
            total_results = sum_results(detections)
            totalAP = np.mean(total_results.precisions)
            total_average_precision.append(totalAP)
            # print('Min Box Area: {} --> {}'.format(min_box_area, total_results.values))

    total_precisions.append(total_average_precision)

pr_total = BaseBarPlot(x_data=[areas] * len(total_precisions),
                       y_data=total_precisions,
                       y_label='Average Precision',
                       x_label='Box Area',
                       colors=['blue', 'red', 'green'],
                       legend=names,
                       title='',
                       line_style=['-x'] * len(total_precisions),
                       width=0.01)

pr_total.show()
