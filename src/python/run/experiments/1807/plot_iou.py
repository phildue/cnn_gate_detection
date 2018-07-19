import os

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
models = [
    'baseline104x104-13x13+9layers',
    # 'baseline208x208-13x13+9layers',
    'baseline416x416-13x13+9layers',
    # 'baseline52x52-13x13+9layers',
    'bottleneck416x416-13x13+9layers',
    # 'bottleneck_narrow416x416-13x13+9layers',
    # 'bottleneck_narrow_strides416x416-13x13+9layers',
    # 'combined208x208-13x13+13layers',
    'grayscale416x416-13x13+9layers',
    # 'narrow416x416-13x13+9layers',
    # 'narrow_strides416x416-13x13+9layers',
    'narrow_strides_late_bottleneck416x416-13x13+9layers',
    #    'strides2416x416-13x13+9layers',
]

names = [
    'baseline104x104',
    'baseline416x416',
    'bottleneck416x416',
    'grayscale416x416',
    'bottleneck_large_strides416x416',
]
for model in models:
    total_average_precision = []
    for iou_thresh in [0.4, 0.6, 0.8]:
        for min_box_area in [0.001]:
            results = load_file(
                'out/1807/' + model + '/test/range_iou{}-area{}_result_metric.pkl'.format(iou_thresh,
                                                                                          min_box_area))
            detections = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
            total_results = sum_results(detections)
            meanAPtotal = np.mean(total_results.precisions[1:])
            total_average_precision.append(meanAPtotal)
            print('Min Box Area: {} --> {}'.format(min_box_area, total_results.values))

    total_precisions.append(total_average_precision)

pr_total = BaseMultiPlot(x_data=[[0.4, 0.6, 0.8]] * len(total_precisions),
                         y_data=total_precisions,
                         y_label='Average Precision',
                         x_label='Intersection Over Union Threshold',
                         y_lim=(0, 1.0),
                         legend=names,
                         title='Total',
                         line_style=['--x'] * len(total_precisions),
                         x_res=None)

pr_total.show()
