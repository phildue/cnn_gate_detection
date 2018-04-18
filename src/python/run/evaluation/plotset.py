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

result_path = 'logs/gatev0_industrial/results/'
result_file = 'industrial_room--38.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result_sum = ResultByConfidence(detection_result[0])
for d in detection_result[1:]:
    detection_result_sum = detection_result_sum + ResultByConfidence(d)

print("True Positives", detection_result_sum.results[0.9].true_positives)
print("Total Predictions",
      detection_result_sum.results[0.9].true_positives + detection_result_sum.results[0.7].false_positives)

mean_pr, mean_re = average_precision_recall([detection_result_sum])

BasePlot(y_data=mean_pr, x_data=mean_re).show()
