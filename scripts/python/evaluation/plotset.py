import numpy as np

from backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from backend.visuals.plots.BasePlot import BasePlot
from frontend.evaluation.ResultsByConfidence import ResultByConfidence
from workdir import work_dir

work_dir()

from fileaccess.utils import load
from frontend.evaluation.EvaluatorPrecisionRecall import EvaluatorPrecisionRecall
from backend.visuals.plots.PlotPrecisionRecall import PlotPrecisionRecall


def mean_avg_prec(results):
    detection_result = results['results']['MetricDetection']
    detection_result = [ResultByConfidence(d) for d in detection_result]
    precision = np.zeros((len(detection_result), 11))
    recall = np.zeros((len(detection_result), 11))
    for i, result in enumerate(detection_result):
        precision[i], recall[i] = EvaluatorPrecisionRecall.interp(result)

    mean_pr = np.mean(precision, 0)
    mean_rec = np.mean(recall, 0)
    return mean_pr, mean_rec


result_path = 'logs/yolov2_10k/set_aligned/'
result_file = 'result_set_aligned.pkl'
results = load(result_path + result_file)

mean_pr_10k, mean_rec_10k = mean_avg_prec(results)

result_path = 'logs/yolov2_25k/set_aligned/'
result_file = 'result_set_aligned.pkl'
results = load(result_path + result_file)

mean_pr_25k, mean_rec_25k = mean_avg_prec(results)

result_path = 'logs/yolov2_50k/set_aligned/'
result_file = 'result_set_aligned.pkl'
results = load(result_path + result_file)

mean_pr_50k, mean_rec_50k = mean_avg_prec(results)

BaseMultiPlot([mean_rec_10k, mean_rec_25k, mean_rec_50k], [mean_pr_10k, mean_pr_25k, mean_pr_50k],
              legend=['10k', '25k', '50k'], x_label='recall', y_label='precision', title='PR-Yolo').show()
# localization_error = results['MetricLocalization']
#
# result_mat = np.vstack([r[0.1] for r in localization_error if r[0.1] is not None])
#
# print(np.mean(result_mat, axis=0))
# print(np.std(result_mat, axis=0))
