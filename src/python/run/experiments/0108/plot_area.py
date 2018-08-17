import os

from modelzoo.backend.visuals.plots.BaseBarPlot import BaseBarPlot
from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
import numpy as np

cd_work()
work_dir = 'out/'

# models = [name for name in os.listdir(work_dir)]
models = [
    '2507/mavnet208x208-13x13+9layers',
    '0108/mavnet208x208-13x13+9layers',
    '0108/multiscale2208x208',
    '0108/multiscale208x208',

]
names = [
    'Prediction at 7x7',
    'Prediction at 13x13',
    'Prediction at 13x13 and 7x7',
    'Prediction at 13x13, 7x7, 3x3 and 1x1',

]
areas = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
legends = []
aps = []
xs = []
linestyles = []
iou_thresh = 0.6
for model in models:
    aps_model = []
    xs_model = []
    for i in range(len(areas) - 1):
        min_box_area = areas[i]
        max_box_area = areas[i + 1]
        results = load_file(
            work_dir + model + '/test/total_iou{}_range{}-{}.pkl'.format(iou_thresh,
                                                                         min_box_area, max_box_area))
        detections = sum_results([ResultByConfidence(r) for r in results['results']['MetricDetection']])
        totalAP_area = np.mean(detections.precisions)
        aps_model.append(totalAP_area)
        linestyles.append('-x')
        xs_model.append(np.sqrt(min_box_area))

        # print('Min Box Area: {} --> {}'.format(min_box_area, total_results.values))
    xs.append(xs_model)
    legends.append([str(x) for x in xs_model])
    aps.append(aps_model)

pr_total = BaseBarPlot(x_data=xs,
                       y_data=aps,
                       y_label='Average Precision at an IoU of {}'.format(iou_thresh),
                       x_label='Box size relative to image size',
                       colors=[(0.5, 0.5, 0.5, 0.6),
                               (0.5, 0, 0.5, 0.6), 'blue', 'red', 'green', 'yellow', 'black', 'magenta', 'cyan',
                               'white'],
                       legend=names,
                       names=['0.01-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.25', '0.25-0.5', '0.5-1.0'],
                       title='',
                       line_style=linestyles,
                       width=0.01)

pr_total.show()
