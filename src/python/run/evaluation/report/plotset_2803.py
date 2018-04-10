import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()


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


def precision_recall(results):
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    result_sum = detection_result[0]
    for i in range(1,len(detection_result)):
        result_sum += detection_result[i]

    print(result_sum.results[0.1].true_positives)
    precision, recall = interp(result_sum)
    return precision, recall


# for i, f in enumerate(results['image_files']):
#     label_true = results['labels_true'][i]
#     label_pred = results['labels_pred'][i][0.3]
#     path = '/run/user/1000/gvfs/sftp:host=student-linux.tudelft.nl/'
#     img = imread(path + f,'bgr')
#     show(img, labels=[label_true, label_pred], colors=[(0, 255, 0), (255, 0, 0)])
#
#
# result_path = 'logs/tiny_bebop_nodistort/results/'
# result_file = 'cyberzoo--21.pkl'
# results = load_file(result_path + result_file)
#
# mean_pr_nodistort, mean_rec_nodistort = precision_recall(results)
#
# result_path = 'logs/tiny_bebop_distort/results/'
# result_file = 'cyberzoo--14.pkl'
# results = load_file(result_path + result_file)
# mean_pr_distort, mean_rec_distort = precision_recall(results)
#
# result_path = 'logs/tiny_bebop_merge/results/'
# result_file = 'cyberzoo--39.pkl'
# results = load_file(result_path + result_file)
# mean_pr_merge, mean_rec_merge = precision_recall(results)
#
# result_path = 'logs/tiny_bebop_google_merge/results/'
# result_file = 'cyberzoo--18.pkl'
# results = load_file(result_path + result_file)
# mean_pr_google, mean_rec_google = precision_recall(results)
#
# BaseMultiPlot(y_data=[mean_pr_distort, mean_pr_merge, mean_pr_nodistort, mean_pr_google],
#               x_data=[mean_rec_distort, mean_rec_merge, mean_rec_nodistort, mean_rec_google],
#               y_label='Precision', x_label='Recall',
#               legend=['Distorted', 'Cyberzoo Background', 'Not Distorted','Cyberzoo+Industrial'],
#               title='Tiny Yolo on mavset').show()
#
#
# result_path = 'logs/v2_bebop_nodistort/results/'
# result_file = 'cyberzoo--22.pkl'
# results = load_file(result_path + result_file)
#
# mean_pr_nodistort, mean_rec_nodistort = precision_recall(results)
#
# result_path = 'logs/v2_bebop_distort/results/'
# result_file = 'cyberzoo--21.pkl'
# results = load_file(result_path + result_file)
# mean_pr_distort, mean_rec_distort = precision_recall(results)
#
# result_path = 'logs/v2_bebop_merge/results/'
# result_file = 'cyberzoo--36.pkl'
# results = load_file(result_path + result_file)
# mean_pr_merge, mean_rec_merge = precision_recall(results)
#
# result_path = 'logs/v2_bebop_google_merge/results/'
# result_file = 'cyberzoo--17.pkl'
# results = load_file(result_path + result_file)
# mean_pr_google, mean_rec_google = precision_recall(results)
#
# BaseMultiPlot(y_data=[mean_pr_distort, mean_pr_merge, mean_pr_nodistort, mean_pr_google],
#               x_data=[mean_rec_distort, mean_rec_merge, mean_rec_nodistort, mean_rec_google],
#               y_label='Precision', x_label='Recall',
#               legend=['Distorted', 'Cyberzoo Background', 'Not Distorted','Cyberzoo+Industrial'],
#               title='YoloV2 on mavset').show()


result_path = 'logs/v2_bebop_nodistort/2803/'
result_file = 'metric_result_2803.pkl'
results = load_file(result_path + result_file)
print(results)
mean_pr_nodistort, mean_rec_nodistort = precision_recall(results)

result_path = 'logs/v2_bebop_distort/2803/'
result_file = 'metric_result_2803.pkl'
results = load_file(result_path + result_file)
mean_pr_distort, mean_rec_distort = precision_recall(results)

result_path = 'logs/v2_bebop_merge/2803/'
result_file = 'metric_result_2803.pkl'
results = load_file(result_path + result_file)
mean_pr_merge, mean_rec_merge = precision_recall(results)

result_path = 'logs/v2_bebop_google_merge/2803/'
result_file = 'metric_result_2803.pkl'
results = load_file(result_path + result_file)
mean_pr_google_merge, mean_rec_google_merge = precision_recall(results)

BaseMultiPlot(y_data=[mean_pr_distort, mean_pr_merge, mean_pr_nodistort,mean_pr_google_merge],
              x_data=[mean_rec_distort, mean_rec_merge, mean_rec_nodistort,mean_rec_google_merge],
              y_label='Precision', x_label='Recall',
              legend=['Distorted', 'Cyberzoo Background', 'Not Distorted','Cyberzoo + Industrial'],
              title='YoloV2 on ethset').show()

# result_path = 'logs/tiny_bebop_nodistort/2803/'
# result_file = 'metric_result_2803.pkl'
# results = load_file(result_path + result_file)
#
# mean_pr_nodistort, mean_rec_nodistort = precision_recall(results)
#
# result_path = 'logs/tiny_bebop_distort/2803/'
# result_file = 'metric_result_2803.pkl'
# results = load_file(result_path + result_file)
# mean_pr_distort, mean_rec_distort = precision_recall(results)
#
# result_path = 'logs/tiny_bebop_merge/2803/'
# result_file = 'metric_result_2803.pkl'
# results = load_file(result_path + result_file)
# mean_pr_merge, mean_rec_merge = precision_recall(results)
#
# result_path = 'logs/tiny_bebop_google_merge/2803/'
# result_file = 'metric_result_2803.pkl'
# results = load_file(result_path + result_file)
# mean_pr_google_merge, mean_rec_google_merge = precision_recall(results)
#
# BaseMultiPlot(y_data=[mean_pr_distort, mean_pr_merge, mean_pr_nodistort,mean_pr_google_merge],
#               x_data=[mean_rec_distort, mean_rec_merge, mean_rec_nodistort,mean_rec_google_merge],
#               y_label='Precision', x_label='Recall',
#               legend=['Distorted', 'Cyberzoo Background', 'Not Distorted','Cyberzoo + Industrial'],
#               title='TinyYolo on ethset').show()

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
