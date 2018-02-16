import numpy as np

from workdir import work_dir

work_dir()

from fileaccess.utils import load
from evaluation.EvaluatorPrecisionRecall import EvaluatorPrecisionRecall
from backend.plots.PlotPrecisionRecall import PlotPrecisionRecall
from evaluation.ResultsByConfidence import ResultByConfidence

result_path = 'logs/yolov2_50k/set_aligned/'
result_file = 'result_set_aligned.pkl'
results = load(result_path + result_file)

detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]

# precision = np.zeros((len(detection_result), 11))
# recall = np.zeros((len(detection_result), 11))
# for i, result in enumerate(detection_result):
#     precision[i], recall[i] = EvaluatorPrecisionRecall.interp(detection_result)
#
# mean_pr = np.mean(precision, 0)
# mean_rec = np.mean(recall, 0)
# PlotPrecisionRecall(mean_pr, mean_rec).show(block=True)

total = detection_result[0]
for result in detection_result[1:]:
    total += result

precision, recall = EvaluatorPrecisionRecall.interp(total)
PlotPrecisionRecall(precision, recall).show(block=True)


# localization_error = results['MetricLocalization']
#
# result_mat = np.vstack([r[0.1] for r in localization_error if r[0.1] is not None])
#
# print(np.mean(result_mat, axis=0))
# print(np.std(result_mat, axis=0))
