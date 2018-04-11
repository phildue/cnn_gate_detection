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

result_path = 'logs/tiny_airsim/results/'
result_file = 'industrial_room--80.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_pr_tiny, mean_rec_tiny = average_precision_recall(detection_result)

result_path = 'logs/tiny_airsim_cats/results/'
result_file = 'industrial_room--80.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_pr_tiny_cats, mean_rec_tiny_cats = average_precision_recall(detection_result)

BaseMultiPlot(y_data=[mean_pr_tiny_cats, mean_pr_tiny],
              x_data=[mean_rec_tiny_cats, mean_rec_tiny],
              y_label='Precision', x_label='Recall',
              legend=['Cats', 'Gates'],
              title='Tiny Yolo on Industrial Room').show(False)

result_path = 'logs/v2_airsim/results/'
result_file = 'industrial_room--40.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_pr_tiny, mean_rec_tiny = average_precision_recall(detection_result)

result_path = 'logs/v2_airsim_cats/results/'
result_file = 'industrial_room--40.pkl'
results = load_file(result_path + result_file)
detection_result = results['results']['MetricDetection']
detection_result = [ResultByConfidence(d) for d in detection_result]
mean_pr_tiny_cats, mean_rec_tiny_cats = average_precision_recall(detection_result)

BaseMultiPlot(y_data=[mean_pr_tiny_cats, mean_pr_tiny],
              x_data=[mean_rec_tiny_cats, mean_rec_tiny],
              y_label='Precision', x_label='Recall',
              legend=['Cats', 'Gates'],
              title='YoloV2 on Industrial Room').show()
