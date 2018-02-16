import numpy as np

from workdir import work_dir

work_dir()

from fileaccess.utils import load
from evaluation.EvaluatorPrecisionRecall import EvaluatorPrecisionRecall
from backend.plots.PlotPrecisionRecall import PlotPrecisionRecall
from evaluation.ResultsByConfidence import ResultByConfidence

result_path = 'logs/yolo-noaug/set04/'
result_file = 'result_set04.pkl'
results = load(result_path + result_file)

detection_result = results['MetricDetection']
detection_result = [ResultByConfidence(d[0]) for d in detection_result]

precision = np.zeros((len(detection_result), 11))
recall = np.zeros((len(detection_result), 11))
for i, result in enumerate(detection_result):
    precision[i], recall[i] = EvaluatorPrecisionRecall.interp(result)

mean_pr = np.mean(precision, 0)
mean_rec = np.mean(recall, 0)
PlotPrecisionRecall(mean_pr, mean_rec).show(block=True)


# localization_error = results['MetricLocalization']
#
# result_mat = np.vstack([r[0.1] for r in localization_error if r[0.1] is not None])
#
# print(np.mean(result_mat, axis=0))
# print(np.std(result_mat, axis=0))
