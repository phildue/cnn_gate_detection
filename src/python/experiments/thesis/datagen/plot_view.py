import matplotlib.pyplot as plt
import numpy as np

from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_dronemodel416x416',
    # 'yolov3_blur416x416',
    # 'yolov3_chromatic416x416',
]

work_dir = 'out/thesis/datagen/'
n_iterations = 1

names = [
    'Random View Points',
    'Flight',
    # 'Flight + Random Blur',
    # 'Flight + Random Chrom',
]
testset = 'iros2018_course_final_simple_17gates'
iou_thresh = 0.6
plt.figure(figsize=(8, 3))
plt.title('Precision - Recall IoU:{}'.format(iou_thresh))
plt.subplot(1, 2, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Results in Virtual Environment")
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

datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]
datasets_names = [
    'Cyberzoo',
    'Basement',
    'Hallway'
]

plt.subplot(1, 2, 2)
plt.title('Results on Real World Datasets'.format(iou_thresh))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim(0.0, 1.1)
precision = np.zeros((3, 11))
recall = np.zeros((3, 11))
for model in models:
    for j, d in enumerate(datasets):
        total_detections = []
        for i in range(n_iterations):
            model_dir = model + '_i0{}'.format(i)
            result_file = work_dir + model_dir + '/test_' + d + '/' + 'results_iou{}.pkl'.format(iou_thresh)
            if "snake" in model:
                result_file = work_dir + model + '{}_boxes{}-{}_iou{}_i0{}.pkl'.format(d, 0, 2.0, iou_thresh, i)
            try:
                results = load_file(result_file)
                total_detections.append(sum_results(results['results']))
            except FileNotFoundError:
                continue

        m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
        precision[j] = m_p
        recall[j] = m_r
    plt.plot(np.mean(recall, 0), np.mean(precision, 0), 'x--')
plt.legend(names)

plt.show(True)
