import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import work_dir

work_dir()


def interp(results: ResultByConfidence, recall_levels=None):
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


def mean_avg_prec(results):
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    precision = np.zeros((len(detection_result), 11))
    recall = np.zeros((len(detection_result), 11))
    for i, result in enumerate(detection_result):
        precision[i], recall[i] = interp(result)

    mean_pr = np.mean(precision, 0)
    mean_rec = np.mean(recall, 0)
    return mean_pr, mean_rec


result_path = '/run/user/1000/gvfs/sftp:host=student-linux.tudelft.nl/home/nfs/pdurnay/dronevision/logs' \
              '/v2_bebop_distort/results/'
result_file = 'cyberzoo--8.pkl'
results = load_file(result_path + result_file)

# for i, f in enumerate(results['image_files']):
#     label_true = results['labels_true'][i]
#     label_pred = results['labels_pred'][i][0.3]
#     path = '/run/user/1000/gvfs/sftp:host=student-linux.tudelft.nl/'
#     img = imread(path + f,'bgr')
#     show(img, labels=[label_true, label_pred], colors=[(0, 255, 0), (255, 0, 0)])
#
mean_pr_10k, mean_rec_10k = mean_avg_prec(results)
BasePlot(y_data=mean_pr_10k, x_data=mean_rec_10k).show()

# result_path = 'logs/yolov2_25k/set_aligned/'
# result_file = 'result_set_aligned.pkl'
# results = load_file(result_path + result_file)
#
# mean_pr_25k, mean_rec_25k = mean_avg_prec(results)
#
# result_path = 'logs/yolov2_50k/set_aligned/'
# result_file = 'result_set_aligned.pkl'
# results = load_file(result_path + result_file)
#
# mean_pr_50k, mean_rec_50k = mean_avg_prec(results)
#
# BaseMultiPlot([mean_rec_10k, mean_rec_25k, mean_rec_50k], [mean_pr_10k, mean_pr_25k, mean_pr_50k],
#               legend=['10k', '25k', '50k'], x_label='recall', y_label='precision', title='PR-Yolo').show()
# # localization_error = results['MetricLocalization']
# #
# # result_mat = np.vstack([r[0.1] for r in localization_error if r[0.1] is not None])
# #
# print(np.mean(result_mat, axis=0))
# print(np.std(result_mat, axis=0))
