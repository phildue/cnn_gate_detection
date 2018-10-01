import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall, sum_results, mean_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = ['yolov3_cats_uniform416x416',
          'yolov3_gate_uniform416x416']

work_dir = 'out/thesis/datagen/'
n_iterations = 1

names = [
    'Complex - Object',
    'EFO - Object',
]
legends = []
linestyles = ['x--', 'x--']

for iou_thresh in [0.4, 0.6, 0.8]:
    dataset = 'random_iros'
    results = []
    for i in range(n_iterations):
        model_folder = 'yolov3_gate_uniform416x416_i0{}'.format(i)
        model_dir = work_dir + model_folder
        results_file = load_file(
            model_dir + '/' + dataset + '/detections_iou{}.pkl'.format(iou_thresh))
        resultsByConf = [ResultByConfidence(r) for r in results_file['results']['MetricDetection']]
        results.append(sum_results(resultsByConf))
    mean_gate = average_precision_recall(results)
    print('Gate mAP{}: {} +- {}'.format(iou_thresh, np.mean(mean_gate[0]), np.mean(mean_gate[2])))

    dataset = 'iros2018_cats'
    results = []
    for i in range(n_iterations):
        model_folder = 'yolov3_cats_uniform416x416_i0{}'.format(i)
        model_dir = work_dir + model_folder
        results_file = load_file(
            model_dir + '/' + dataset + '/detections_iou{}.pkl'.format(iou_thresh))
        resultsByConf = [ResultByConfidence(r) for r in results_file['results']['MetricDetection']]
        results.append(sum_results(resultsByConf))
    mean_cats = average_precision_recall(results)
    print('Cats mAP{}: {} +- {}'.format(iou_thresh, np.mean(mean_cats[0]), np.mean(mean_cats[2])))
