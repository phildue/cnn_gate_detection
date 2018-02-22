import numpy as np

from frontend.evaluation.ResultsByConfidence import ResultByConfidence
from workdir import work_dir

work_dir()

from fileaccess.utils import load
from frontend.evaluation.EvaluatorPrecisionRecall import EvaluatorPrecisionRecall
from backend.visuals.plots.PlotPrecisionRecall import PlotPrecisionRecall

result_path = 'logs/yolov2_25k/set_aligned/'
result_file = 'result_set_aligned.pkl'
results = load(result_path + result_file)

detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]

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
