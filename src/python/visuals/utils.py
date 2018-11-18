import numpy as np

from evaluation import ResultByConfidence
from evaluation.utils import avg_pr_per_image_file
from utils.fileaccess.utils import load_file
from visuals import BaseMultiPlot


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
