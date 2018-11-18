import matplotlib.pyplot as plt
import numpy as np

from evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    'yolov3_gate_realbg416x416',
    'yolov3_gate_uniform416x416',
    'yolov3_gate_dronemodel416x416',
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_mixed416x416',
    'yolov3_hsv416x416',
    'yolov3_blur416x416',
    'yolov3_chromatic416x416',
    'yolov3_exposure416x416',
]

work_dir = 'out/thesis/datagen/'
n_iterations = 1

names = [
    'Real Backgrounds',
    'Uniform Backgrounds',
    'Flight Images',
    'Various Environments',
    'Real + Various',
    'Color Variations',
    'Blur',
    'Chromatic',
    'Exposure',
]
testset = 'iros2018_course_final_simple_17gates'
# testset = 'jevois_hallway'
legends = []
linestyles = ['x--'] * len(names)
iou_thresh = 0.6
min_box_area = 0.01
max_box_area = 2.0
ar = [4.0]
mean_recalls = []
mean_precisions = []
std_precisions = []

# testset = 'jevois_hallway'
iou_thresh = 0.4
plt.figure(figsize=(8, 3))
plt.title('Precision - Recall IoU:{}'.format(iou_thresh))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Simu")
plt.ylim(0.0, 1.1)
for model in models:
    total_detections = []
    mean_detections = []
    for i in range(n_iterations):
        model_dir = model + '_i0{}'.format(i)
        result_file = work_dir + model_dir + '/test_' + testset + '/' + 'results_iou{}.pkl'.format(iou_thresh)
        if "snake" in model:
            result_file = 'out/thesis/snake/test_{}_results_iou{}_{}.pkl'.format(testset, iou_thresh, i)
        try:
            results = load_file(result_file)
            total_detections.append(sum_results(results['results']))
        except FileNotFoundError:
            continue

    m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
    print(m_p)
    print('{}:  map{}: {}'.format(model, iou_thresh, np.mean(m_p)))
    plt.plot(m_r, m_p, 'x--')

plt.legend(names)
plt.show(True)
