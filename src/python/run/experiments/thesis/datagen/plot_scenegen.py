import numpy as np

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall, sum_results, mean_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = ['datagen/yolov3_gate_realbg416x416',
          'datagen/yolov3_gate416x416',
          'datagen/yolov3_gate_varioussim416x416',
          'snake/']

work_dir = 'out/thesis/'
n_iterations = 1

names = [
    'Real Backgrounds',
    'Basement Environment',
    'Various Environments',
    'Snake Gate'
]
testset = 'iros2018_course_final_simple_17gates'
legends = []
linestyles = ['x--', 'x--', 'x--', 'o--']
iou_thresh = 0.8
mean_total_detections = []
mean_average_detections = []
average_precisions = []
average_recalls = []
for model in models:
    total_detections = []
    mean_detections = []
    avg_precision_recall = []
    for i in range(n_iterations):
        model_dir = model + '_i0{}'.format(i)
        result_file = work_dir + model_dir + '/test/scenegen_results_{}_iou{}.pkl'.format(testset, iou_thresh)
        if "snake" in model:
            result_file = work_dir + model + '{}_iou{}.pkl'.format(testset, iou_thresh)
        aps_model = []
        xs_model = []
        results = load_file(result_file)
        resultsByConf = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
        avg_precision_recall.append(average_precision_recall(resultsByConf))
        total_detections.append(sum_results(resultsByConf))
        mean_detections.append(mean_results(resultsByConf))

    mean_total_detections.append(mean_results(total_detections))
    mean_average_detections.append(mean_results(mean_detections))

    average_precision = np.zeros((11,))
    average_recall = np.zeros((11,))
    for p, r in avg_precision_recall:
        average_precision += p / len(avg_precision_recall)
        average_recall += r / len(avg_precision_recall)
    average_precisions.append(average_precision)
    average_recalls.append(average_recall)

pr_total = BaseMultiPlot(x_data=[x.recalls for x in mean_total_detections],
                         y_data=[x.precisions for x in mean_total_detections],
                         y_label='Precision',
                         x_label='Recall',
                         legend=names,
                         y_lim=(0.0, 2.0),
                         title='Mean Total Precision at an IoU of {}'.format(iou_thresh),
                         line_style=linestyles)

pr_mean = BaseMultiPlot(x_data=average_recalls,
                        y_data=average_precisions,
                        y_label='Precision',
                        x_label='Recall',
                        y_lim=(0.0, 2.0),
                        legend=names,
                        title='Mean Average Precision at an IoU of {}'.format(iou_thresh),
                        line_style=linestyles)
pr_total.show(False)
pr_mean.show()
