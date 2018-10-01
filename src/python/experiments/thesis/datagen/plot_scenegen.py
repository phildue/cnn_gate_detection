from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
import numpy as np

cd_work()
models = ['datagen/yolov3_gate_realbg416x416',
          'datagen/yolov3_gate416x416',
          'datagen/yolov3_gate_varioussim416x416',
          'datagen/yolov3_gate_dronemodel416x416',
          # 'snake/',
          'datagen/yolov3_gate_uniform416x416',
          'datagen/yolov3_gate_mixed416x416',
          'datagen/yolov3_gate_pp416x416'
          ]

work_dir = 'out/thesis/'
n_iterations = 5

names = [
    'Real Backgrounds',
    'Basement Environment',
    'Various Environments',
    'Drone Model',
    # 'Snake Gate',
    'Uniform',
    'Real + Sim',
    'Postproc'
]
# testset = 'iros2018_course_final_simple_17gates'
testset = 'iros2018_course_final_simple_17gates'
legends = []
linestyles = ['x--']*len(names)
iou_thresh = 0.4
min_box_area = 0.01
max_box_area = 2.0
ar = [4.0]
mean_recalls = []
mean_precisions = []
std_precisions = []
for model in models:
    total_detections = []
    mean_detections = []
    for i in range(n_iterations):
        model_dir = model + '_i0{}'.format(i)
        result_file = work_dir + model_dir + '/' +'scenegen' + '/' + 'results_{}_boxes{}-{}_iou{}.pkl'.format(testset,
                                                                                                     min_box_area,
                                                                                                     max_box_area,
                                                                                                     iou_thresh)
        if "snake" in model:
            result_file = work_dir + model + '{}_boxes{}-{}_iou{}_i0{}.pkl'.format(testset, 0, 2.0, iou_thresh, i)
        try:
            results = load_file(result_file)
            resultsByConf = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
            total_detections.append(sum_results(resultsByConf))
        except FileNotFoundError:
            continue

    m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
    print(m_p)
    print('{}:  map{}: {}'.format(model, iou_thresh, np.mean(m_p)))
    mean_recalls.append(m_r)
    mean_precisions.append(m_p)
    std_precisions.append(std_p)

pr_total = BaseMultiPlot(x_data=mean_recalls,
                         y_data=mean_precisions,
                         y_label='Precision',
                         x_label='Recall',
                         legend=names,
                         y_lim=(0.0, 1.1),
                         title='Mean Total Precision at an IoU of {}'.format(iou_thresh),
                         line_style=linestyles)

pr_total.show(True)
