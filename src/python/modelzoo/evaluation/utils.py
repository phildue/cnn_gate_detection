import numpy as np

from modelzoo.evaluation.DetectionResult import DetectionResult
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from utils.fileaccess.utils import load_file


def sum_results(detection_results: [ResultByConfidence]):
    """
    Sums list
    :param detection_results: list of results per image
    :return: sum
    """
    result_sum = detection_results[0]
    for d in detection_results[1:]:
        result_sum += d

    return result_sum


def mean_results(detection_results: [ResultByConfidence]):
    confidence = np.round(np.linspace(0, 1.0, 11), 2)

    mean_r = {}
    for j, c in enumerate(confidence):
        result_mat = np.zeros((len(detection_results), 3))

        for i, result in enumerate(detection_results):
            result_mat[i, 0] = result.results[c].true_positives
            result_mat[i, 1] = result.results[c].false_positives
            result_mat[i, 2] = result.results[c].false_negatives

        mean = np.mean(result_mat, 0)
        mean_r[c] = DetectionResult(mean[0], mean[1], mean[2])

    return ResultByConfidence(mean_r)


def average_precision_recall(detection_results: [ResultByConfidence], recall_levels=None):
    """
    Calculates average precision recall with interpolation. According to mAP of Pascal VOC metric.
    :param detection_results: list of results for each image
    :return: tensor(1,11): mean precision, tensor(1,11) mean recall
    """
    if recall_levels is None:
        recall_levels = np.linspace(0, 1.0, 11)
    precision = np.zeros((len(detection_results), len(recall_levels)))
    recall = np.zeros((len(detection_results), len(recall_levels)))
    for i, result in enumerate(detection_results):
        skip = False
        for c in result.results.keys():
            if (result.results[c].false_positives < 0 or
                    result.results[c].true_positives < 0 or
                    result.results[c].false_negatives < 0):
                skip = True
                print('Warning weird numbers')
        if skip: continue
        precision[i], recall[i] = interpolate(result, recall_levels)

    mean_pr = np.mean(precision, 0)
    std_pr = np.std(precision, 0)

    mean_rec = np.mean(recall, 0)
    std_rec = np.std(recall, 0)
    return mean_pr, mean_rec, std_pr, std_rec


def interpolate(results: ResultByConfidence, recall_levels=None):
    if recall_levels is None:
        recall_levels = np.linspace(0, 1.0, 11)

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
