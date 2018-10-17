import numpy as np
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence

from modelzoo.evaluation.utils import sum_results
from modelzoo.visuals.plots.BaseMultiPlot import BaseMultiPlot
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()

legends = []
total_precisions = []
linestyle = ['-.', '-*', '-x', '-o', '--']
models = [
    'baseline104x104-13x13+9layers',
    'baseline208x208-13x13+9layers',
    'baseline416x416-13x13+9layers',
    'baseline52x52-13x13+9layers',
    'bottleneck416x416-13x13+9layers',
    # 'bottleneck_narrow416x416-13x13+9layers',
    # 'bottleneck_narrow_strides416x416-13x13+9layers',
    # 'combined208x208-13x13+13layers',
    # 'grayscale416x416-13x13+9layers',
    # 'narrow416x416-13x13+9layers',
    # 'narrow_strides416x416-13x13+9layers',
    # 'narrow_strides_late_bottleneck416x416-13x13+9layers',
    #    'strides2416x416-13x13+9layers',
]

names = [
    'baseline104x104',
    'baseline208x208',
    'baseline416x416',
    'baseline52x52',
]
for model in models:
    total_average_precision = []
    for iou_thresh in [0.4]:
        for min_box_area in [0.001, 0.025, 0.05, 0.1, 0.15, 0.25]:
            results = load_file(
                'out/1807/' + model + '/test/range_iou{}-area{}_result_metric.pkl'.format(iou_thresh,
                                                                                                        min_box_area))
            detections = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
            total_results = sum_results(detections)
            meanAPtotal = np.mean(total_results.precisions)
            total_average_precision.append(meanAPtotal)
            print('Min Box Area: {} --> {}'.format(min_box_area, total_results.values))

    total_precisions.append(total_average_precision)

pr_total = BaseMultiPlot(x_data=[[0.001, 0.025, 0.05, 0.1, 0.15, 0.25]] * len(total_precisions),
                         y_data=total_precisions,
                         y_label='Average Precision',
                         x_label='Min Box Area',
                         y_lim=(0, 1.0),
                         legend=names,
                         title='',
                         line_style=['-x'] * len(total_precisions),
                         x_res=None)

pr_total.show()
