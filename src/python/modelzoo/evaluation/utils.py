import numpy as np

from modelzoo.evaluation.DetectionResult import DetectionResult
from utils.fileaccess.utils import load_file


def sum_results(detection_results: [DetectionResult]):
    """
    Sums list
    :param detection_results: list of results per image
    :return: sum
    """
    result_sum = detection_results[0]
    for d in detection_results[1:]:
        result_sum += d

    return result_sum


def average_precision_recall(detection_results: [DetectionResult], recall_levels=None):
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
        precision_raw = result.precision_conf
        recall_raw = result.recall_conf

        precision[i], recall[i] = interpolate(precision_raw, recall_raw, recall_levels)

    mean_pr = np.mean(precision, 0)
    std_pr = np.std(precision, 0)

    mean_rec = np.mean(recall, 0)
    std_rec = np.std(recall, 0)
    return mean_pr, mean_rec, std_pr, std_rec


def interpolate(precision_raw, recall_raw, recall_levels=None):
    if recall_levels is None:
        recall_levels = np.linspace(0, 1.0, 11)

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



