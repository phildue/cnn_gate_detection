import os

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
import numpy as np

cd_work()

legends = []
mean_precisions = []
total_precision = []
linestyle = ['-.', '-*', '-x', '-o', '--']
for model in [
    # 'baseline104x104-13x13+9layers',
    #       'baseline208x208-13x13+9layers',
    #       'baseline416x416-13x13+9layers',
    #       'baseline52x52-13x13+9layers',
    #       'bottleneck416x416-13x13+9layers',
    #       'bottleneck_narrow416x416-13x13+9layers',
    # 'bottleneck_narrow_strides416x416-13x13+9layers',
    # 'combined208x208-13x13+13layers',
    # 'grayscale416x416-13x13+9layers',
    # 'mobilenetV1416x416-13x13+9layers',
    # 'narrow416x416-13x13+9layers',
    # 'narrow_strides416x416-13x13+9layers',
    # 'narrow_strides_late_bottleneck416x416-13x13+9layers',
    'strides2416x416-13x13+9layers',
    # 'strides416x416-13x13+9layers'
]:
    for iou_thresh in [0.4]:
        for min_box_area in [0.001, 0.025, 0.05, 0.1, 0.25]:
            results = load_file(
                'out/1807/' + model + '/test/test_iou{}-area{}_result_metric.pkl'.format(iou_thresh, min_box_area))
            detections = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
            mean_pr, mean_recall = average_precision_recall(detections)
            total_results = sum_results(detections)

            meanAP = np.mean(mean_pr)
            meanAPtotal = np.mean(total_results.precisions[1:])
            mean_precisions.append(meanAP)
            total_precision.append(meanAPtotal)

pr_img = BaseMultiPlot(x_data=[[0.001, 0.025, 0.05, 0.1, 0.25]],
                       y_data=[mean_precisions],
                       y_label='Average Precision',
                       x_label='Min Box Area',
                       y_lim=(0, 1.0),
                       legend=['strides2416x416-13x13+9layers', ],
                       title='Per Image',
                       # line_style=linestyle,
                       x_res=None)

pr_total = BaseMultiPlot(x_data=[[0.001, 0.025, 0.05, 0.1, 0.25]],
                         y_data=[total_precision],
                         y_label='Average Precision',
                         x_label='Min Box Area',
                         y_lim=(0, 1.0),
                         legend=['strides2416x416-13x13+9layers'],
                         title='Total',
                         #  line_style=linestyle,
                         x_res=None)

pr_img.show(False)
pr_total.show()
