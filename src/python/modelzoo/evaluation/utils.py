import numpy as np

from modelzoo.evaluation import ResultsByConfidence


def average_precision_recall(detection_results):
    precision = np.zeros((len(detection_results), 11))
    recall = np.zeros((len(detection_results), 11))
    for i, result in enumerate(detection_results):
        precision[i], recall[i] = interpolate(result)

    mean_pr = np.mean(precision, 0)
    mean_rec = np.mean(recall, 0)
    return mean_pr, mean_rec


def interpolate(results: ResultsByConfidence, recall_levels=None):
    if recall_levels is None:
        recall_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    sorted_results = results.values
    precision_raw = np.zeros((1, len(sorted_results)))
    recall_raw = np.zeros((1, len(sorted_results)))
    for i, r in enumerate(sorted_results):
        precision_raw[0, i] = r.precision
        recall_raw[0, i] = r.recall

    precision = np.zeros(shape=(len(recall_levels)))
    for i, r in enumerate(recall_levels):
        try:
            idx = np.where(recall_raw[0, :] > r)[0][0]
            precision[i] = np.max(precision_raw[0, idx:])
        except IndexError:
            precision[i] = 0
    return precision, recall_levels.T