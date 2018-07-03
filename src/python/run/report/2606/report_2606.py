from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
import numpy as np


def load_mean_pr(netname, img_res, grid, layers, filters, old=True):
    if old:
        if filters == 16:
            folder_name = '{}{}x{}-{}layers'.format(netname, img_res[0], img_res[1], layers)
        else:
            folder_name = '{}{}x{}-{}layers-{}filters'.format(netname, img_res[0], img_res[1], layers, filters)
    else:
        folder_name = '{}{}x{}-{}x{}+{}layers+{}filters'.format(netname, img_res[0], img_res[1], grid[0],
                                                                grid[1], layers, filters)

    content = load_file('out/2606/' + folder_name + '/results/result_0.4.pkl')
    mean_pr, mean_rec = average_precision_recall(content['results']['MetricDetection'])
    return mean_pr, mean_rec


cd_work()
# layers = np.array([l for l in range(2, 9)])
# cropnet5216 = np.array([load_mean_pr('cropnet', (52, 52), (13, 13), l, 16) for l in layers])
# cropnet5232 = np.array([load_val_acc('cropnet', (52, 52), (13, 13), l, 32) for l in layers])
# cropnet5248 = np.array([load_val_acc('cropnet', (52, 52), (13, 13), l, 48) for l in layers])
# cropnet10416 = np.array([load_val_acc('cropnet', (104, 104), (13, 13), l, 16) for l in layers])
# cropnet416643 = np.array([load_val_acc('cropnet', (416, 416), (3, 3), l, 64, False) for l in [3, 5, 7, 9, 11]])
#
# crpnetplot = BaseMultiPlot(
#     y_data=[cropnet5216, cropnet5232, cropnet5248, cropnet10416, cropnet416643], y_label='Validation Accuracy',
#     x_data=[layers, layers, layers, layers, np.array([3, 5, 7, 9, 11])], x_label='Layers',
#     legend=['52x52->13x13-16filters', '52x52->13x13-32filters', '52x52->13x13-48filters',
#             '104x104->13x13-16filters',
#             '416x416->3x3-16filters'],
#     line_style=['--x', '--x', '--x', '--*', '--o', '--o'],
#     y_lim=(0, 0.6)

# )
# crpnetplot.show(False)

"""
Refnet

"""
# refnet416 = load_mean_pr('refnet', (52, 52), (3, 3), 4, 16, False)
refnet432 = load_mean_pr('refnet', (52, 52), (3, 3), 4, 32, False)
# refnet464 = load_mean_pr('refnet', (52, 52), (3, 3), 4, 64, False)
#
# refnet616 = load_mean_pr('refnet', (52, 52), (3, 3), 6, 16, False)
# refnet632 = load_mean_pr('refnet', (52, 52), (3, 3), 6, 32, False)
# refnet664 = load_mean_pr('refnet', (52, 52), (3, 3), 6, 64, False)
#
# refnetplot = BaseMultiPlot(
#     y_data=[refnet416[0], refnet432[0], refnet464[0], refnet616[0], refnet632[0], refnet664[0]], y_label='Precision',
#     x_data=[refnet416[1], refnet432[1], refnet464[1], refnet616[1], refnet632[1], refnet664[1]], x_label='Recall',
#     legend=['4 l 16f', '4 l 32f', '4 l 64f', '6 l 16f', '6 l 32f', '6 l 64f'],
#     line_style=['--x', '--x', '--x', '--o', '--o', '--o'],
#     y_lim=(0, 1.0)
#
# )
# refnetplot.show()
