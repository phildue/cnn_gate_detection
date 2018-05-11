import numpy as np

from modelzoo.backend.visuals.plots.BaseBarPlot import BaseBarPlot
from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work


def mean_pr_c(result_by_conf):
    precision = np.zeros((len(result_by_conf), 10))
    recall = np.zeros((len(result_by_conf), 10))
    confidence = np.round(np.linspace(0.1, 1.0, 10), 2)

    for i, result in enumerate(result_by_conf):
        for j, c in enumerate(confidence):
            precision[i, j] = result.results[c].precision
            recall[i, j] = result.results[c].recall

    mean_precision = np.mean(precision, 0)
    mean_recall = np.mean(recall, 0)
    return mean_precision, mean_recall, confidence


def avg_pr_file(src_file):
    results = load_file(src_file)
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    mean_pr, mean_rec = average_precision_recall(detection_result)
    return mean_pr, mean_rec


def pr_plot(files, legend, title, line_style=None, y_range=(0.5, 1.0)):
    recalls = []
    precisions = []
    for f in files:
        mean_p, mean_r = avg_pr_file(f)
        recalls.append(mean_r)
        precisions.append(mean_p)

    return BaseMultiPlot(x_data=recalls,
                         y_data=precisions,
                         y_label='Precision',
                         x_label='Recall',
                         y_lim=y_range,
                         legend=legend,
                         title=title,
                         line_style=line_style)


def mean_detections(result_by_conf):
    true_positives = np.zeros((len(result_by_conf), 10))
    false_positives = np.zeros((len(result_by_conf), 10))
    false_negatives = np.zeros((len(result_by_conf), 10))
    n_predictions = np.zeros((len(result_by_conf), 10))
    n_objs = np.zeros((len(result_by_conf), 10))
    confidence = np.round(np.linspace(0.1, 1.0, 10), 2)

    for i, result in enumerate(result_by_conf):
        for j, c in enumerate(confidence):
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
    return mean_tp, mean_fp, mean_fn, mean_pred, mean_objs, confidence


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


def speed_plot(src_files):
    curves = []
    ind = []
    colors = []
    for i, src_file in enumerate(src_files, 1):
        results = load_file(src_file)
        results_pred = results['results_pred']
        results_pp = results['results_pp']
        results_total = np.sum([results_pp, results_pred], 0)
        curves.extend([np.mean(results_pred), np.mean(results_pp), np.mean(results_total)])
        ind.extend([i] * 3)
        colors.extend(['blue', 'red', 'green'])
    return BaseBarPlot(x_data=ind,
                       y_data=curves,
                       colors=None)


cd_work()
speed_plot(['out/gatev9_mixed/speed/result.pkl', 'out/gatev10_mixed/speed/result.pkl']).show()
# pr_daylight_tuning = pr_plot(files=[
#     'out/gatev5_mixed/results/daylight--024.pkl',
#     'out/gatev6-1/results/daylight--017.pkl',
#     'out/gatev7_mixed/results/daylight--022.pkl',
#     'out/gatev8_mixed/results/daylight--020.pkl',
#     'out/gatev9_mixed/results/daylight--011.pkl',
#     'out/gatev10_mixed/results/daylight--011.pkl',
#     'out/gatev11_mixed/results/daylight--008.pkl',
#     'out/gatev12_mixed/results/daylight--008.pkl',
#     'out/gatev13_mixed/results/daylight--008.pkl',
#     'out/tiny_mixed/results/daylight--023.pkl',
#     'out/v2_mixed/results/daylight--019.pkl'
# ],
#     legend=['GateNet5-Mixed',
#             'GateNet6-Mixed',
#             'GateNet7-Mixed',
#             'GateNet8-Mixed',
#             'GateNet9-Mixed',
#             'GateNet10-Mixed',
#             'GateNet11-Mixed',
#             'GateNet12-Mixed',
#             'GateNet13-Mixed',
#             'Tiny-Mixed',
#             'V2-Mixed'
#             ],
#     title='Test on Daylight',
#     line_style=['x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'x-', 'o:', '.-', ],
#     y_range=(0.8, 1.0))
#
# pr_daylight_tuning.show(True)
