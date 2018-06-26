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
        folder_name = '{}{}x{}->{}x{}+{}layers+{}filters'.format(netname, img_res[0], img_res[1], grid[0],
                                                                 grid[1], layers, filters)

    content = load_file('out/' + folder_name + '/results/result_.pkl')
    mean_pr, mean_rec = average_precision_recall(content)
    return np.mean(mean_pr)


cd_work()
layers = np.array([l for l in range(2, 9)])
cropnet5216 = np.array([load_mean_pr('cropnet', (52, 52), (13, 13), l, 16) for l in layers])
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
# layers = np.array([l for l in [4, 6]])
# # refnet52163 = np.array([load_val_acc('refnet', (52, 52), (3, 3), l, 16, False) for l in layers])
# refnet52323 = np.array([load_map('refnet', (52, 52), (3, 3), l, 32, False) for l in layers])
# refnet52643 = np.array([load_map('refnet', (52, 52), (3, 3), l, 64, False) for l in layers])
#
# refnetplot = BaseMultiPlot(
#     y_data=[refnet52323, refnet52643], y_label='Mean Average Precision',
#     x_data=[layers, layers], x_label='Layers',
#     legend=['52x52->3x3-32filters', '52x52->3x3-64filters', ],
#     line_style=['--x', '--x'],
#     y_lim=(0.5, 1.0)
#
# )
# refnetplot.show()
