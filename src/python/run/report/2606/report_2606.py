from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
import numpy as np


def load_result(netname, img_res, grid, layers, filters, old=True, filename='result_0.4.pkl'):
    if old:
        if filters == 16:
            folder_name = '{}{}x{}-{}layers'.format(netname, img_res[0], img_res[1], layers)
        else:
            folder_name = '{}{}x{}-{}layers-{}filters'.format(netname, img_res[0], img_res[1], layers, filters)
    else:
        folder_name = '{}{}x{}-{}x{}+{}layers+{}filters'.format(netname, img_res[0], img_res[1], grid[0],
                                                                grid[1], layers, filters)

    return load_file('out/2606/' + folder_name + '/results/' + filename)


def avg_pr_per_image_file(src_file):
    results = load_file(src_file)
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    mean_pr, mean_rec = average_precision_recall(detection_result)
    return mean_pr, mean_rec


def pr_plot(files, legend, title, line_style=None, y_range=(0, 1.0)):
    recalls = []
    precisions = []
    for f in files:
        mean_p, mean_r = avg_pr_per_image_file(f)
        recalls.append(mean_r)
        precisions.append(mean_p)

    return BaseMultiPlot(x_data=recalls,
                         y_data=precisions,
                         y_label='Precision',
                         x_label='Recall',
                         y_lim=y_range,
                         legend=legend,
                         title=title,
                         line_style=line_style,
                         x_res=None)


def mean_detections(result_by_conf):
    true_positives = np.zeros((len(result_by_conf), 11))
    false_positives = np.zeros((len(result_by_conf), 11))
    false_negatives = np.zeros((len(result_by_conf), 11))
    n_predictions = np.zeros((len(result_by_conf), 11))
    n_objs = np.zeros((len(result_by_conf), 11))

    for i, result in enumerate(result_by_conf):
        for j, c in enumerate(sorted(result.results.keys())):
            true_positives[i, j] = result.results[c].true_positives
            false_positives[i, j] = result.results[c].false_positives
            false_negatives[i, j] = result.results[c].false_negatives
            n_predictions[i, j] = result.results[c].true_positives + result.results[c].false_positives
            n_objs[i, j] = result.results[c].true_positives + result.results[c].false_negatives

    mean_tp = np.mean(true_positives, 0)
    mean_fp = np.mean(false_positives, 0)
    mean_fn = np.mean(false_negatives, 0)
    mean_pred = np.mean(n_predictions, 0)
    mean_objs = np.mean(n_objs, 0)
    return mean_tp, mean_fp, mean_fn, mean_pred, mean_objs, sorted(result_by_conf[0].results.keys())

def detection_plot(src_file, title):
    results = load_file(src_file)
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    mean_tp, mean_fp, mean_fn, mean_pred, mean_objs, confidence = mean_detections(detection_result)
    return BaseMultiPlot(x_data=[confidence] * 5, y_data=[mean_tp, mean_pred, mean_objs],
                         legend=['True Positives', 'Predictions', 'True Objects'],
                         x_label='Confidence',
                         y_label='Predictions',
                         line_style=['.-', '.-', ':'],
                         title=title)


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
