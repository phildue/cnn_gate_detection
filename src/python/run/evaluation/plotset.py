import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()


result_path = 'logs/tiny_bebop_nodistort/results/'
result_file = 'cyberzoo--21.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_pr_nodistort, mean_rec_nodistort = average_precision_recall(detection_result)

result_path = 'logs/tiny_bebop_distort/results/'
result_file = 'cyberzoo--14.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_pr_distort, mean_rec_distort = average_precision_recall(detection_result)

result_path = 'logs/tiny_bebop_merge/results/'
result_file = 'cyberzoo--39.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_pr_merge, mean_rec_merge = average_precision_recall(detection_result)

BaseMultiPlot(y_data=[mean_pr_distort, mean_pr_merge, mean_pr_nodistort],
              x_data=[mean_rec_distort, mean_rec_merge, mean_rec_nodistort],
              y_label='Precision', x_label='Recall',
              legend=['Distorted', 'Cyberzoo Background', 'Not Distorted'],
              title='Tiny Yolo on mavset').show()


result_path = 'logs/v2_bebop_nodistort/results/'
result_file = 'cyberzoo--22.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_pr_nodistort, mean_rec_nodistort = average_precision_recall(results)

result_path = 'logs/v2_bebop_distort/results/'
result_file = 'cyberzoo--21.pkl'
results = load_file(result_path + result_file)
mean_pr_distort, mean_rec_distort = average_precision_recall(results)

result_path = 'logs/v2_bebop_merge/results/'
result_file = 'cyberzoo--36.pkl'
results = load_file(result_path + result_file)
mean_pr_merge, mean_rec_merge = average_precision_recall(results)

BaseMultiPlot(y_data=[mean_pr_distort, mean_pr_merge, mean_pr_nodistort],
              x_data=[mean_rec_distort, mean_rec_merge, mean_rec_nodistort],
              y_label='Precision', x_label='Recall',
              legend=['Distorted', 'Cyberzoo Background', 'Not Distorted'],
              title='YoloV2 on mavset').show()

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
