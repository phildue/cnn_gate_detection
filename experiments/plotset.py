import numpy as np

from workdir import work_dir

work_dir()

from fileaccess.utils import load
from evaluation.EvaluatorPrecisionRecall import EvaluatorPrecisionRecall
from backend.plots.PlotPrecisionRecall import PlotPrecisionRecall
from evaluation.ResultsByConfidence import ResultByConfidence

result_path = 'logs/tinyyolo-noaug/set04/'
result_file = 'result_set04.pkl'
results = load(result_path + result_file)

detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
total = detection_result[0]
for result in detection_result[1:]:
    total = total + result

precision, recall = EvaluatorPrecisionRecall.interp(total)
PlotPrecisionRecall(precision, recall).show(block=True)


# localization_error = results['MetricLocalization']
#
# result_mat = np.vstack([r[0.1] for r in localization_error if r[0.1] is not None])
#
# print(np.mean(result_mat, axis=0))
# print(np.std(result_mat, axis=0))