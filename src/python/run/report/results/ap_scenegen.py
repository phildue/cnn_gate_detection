import numpy as np
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence

from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = ['datagen/yolov3_gate_realbg416x416',
          'datagen/yolov3_gate416x416',
          'datagen/yolov3_gate_varioussim416x416',
          'datagen/yolov3_gate_dronemodel416x416',
          # 'snake/',
          'datagen/yolov3_gate_uniform416x416',
          'datagen/yolov3_gate_mixed416x416'
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
    'Real + Sim'
]
# testset = 'iros2018_course_final_simple_17gates'
testset = 'jevois_cyberzoo'
legends = []
linestyles = ['x--', 'x--', 'x--', 'x--', 'x--', 'x--']
ious = [0.4, 0.6, 0.8]
min_box_area = 0.01
max_box_area = 1.0
for model in models:
    for iou in ious:
        total_detections = []
        for i in range(n_iterations):

            model_dir = model + '_i0{}'.format(i)
            result_file = work_dir + model_dir + '/scenegen/' + 'results_{}_boxes{}-{}_iou{}.pkl'.format(testset,
                                                                                                         min_box_area,
                                                                                                         max_box_area,
                                                                                                         iou)
            if "snake" in model:
                result_file = work_dir + model + '{}_boxes{}-{}_iou{}_i0{}.pkl'.format(testset, 0, 2.0, iou, i)
            try:
                results = load_file(result_file)
                resultsByConf = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
                total_detections.append(sum_results(resultsByConf))
            except FileNotFoundError:
                continue

        m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
        print('{}:  map{}: {}'.format(model, iou, np.mean(m_p)))

