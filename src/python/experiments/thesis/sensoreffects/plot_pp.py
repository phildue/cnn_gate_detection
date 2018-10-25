import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    # 'yolov3_gate_realbg416x416',
    # 'yolov3_gate_uniform416x416',
    # 'yolov3_gate_varioussim416x416',
    # 'yolov3_gate_mixed416x416',
    # 'yolov3_gate_dronemodel416x416',
    # 'yolov3_allgen416x416',
    'old_sensor/yolov3_hsv416x416',
    'old_sensor/yolov3_blur416x416',
    'old_sensor/yolov3_chromatic416x416',
    'old_sensor/yolov3_exposure416x416',
    # 'yolov3_40k416x416',
    'yolov3_allview416x416',
]

work_dir = 'out/thesis/datagen/'
n_iterations = 1

names = [
    # 'Real Backgrounds',
    # 'Uniform Backgrounds',
    # 'Various Environments',
    # 'Real + Various',
    # 'Flight',
    # 'All',
    'HSV',
    'Blur',
    'Chromatic',
    'Exposure',
    # '40k',
    'No Augmentation'
]

testset = 'iros2018_course_final_simple_17gates'
# iou_thresh = 0.6
iou_thresh = [0.4, 0.6, 0.8]
plt.figure(figsize=(8, 3))
plt.title('Precision - Recall IoU:{}'.format(iou_thresh))
plt.subplot(1, 2, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Results in Virtual Environment")
plt.ylim(0.0, 1.1)
frame = pd.DataFrame()
frame['Name'] = pd.Series(names)

for iou in iou_thresh:
    results_on_sim = []
    for m, model in enumerate(models):
        total_detections = []
        mean_detections = []
        for i in range(n_iterations):
            model_dir = model + '_i0{}'.format(i)
            result_file = work_dir + model_dir + '/test_' + testset + '/' + 'results_iou{}.pkl'.format(iou)
            if "snake" in model:
                result_file = 'out/thesis/snake/test_{}_results_iou{}_{}.pkl'.format(testset, iou, i)
            try:
                results = load_file(result_file)
                total_detections.append(sum_results(results['results']))
            except FileNotFoundError:
                continue

        m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
        meanAp = np.mean(m_p)
        errAp = np.mean(std_p)
        results_on_sim.append(np.round(meanAp,2))  # , errAp
        plt.plot(m_r, m_p, 'x--')
    frame['Sim Data' + str(iou)] = pd.Series(results_on_sim)

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
err_p = np.zeros((3, 11))
for iou in iou_thresh:
    results_on_real = []
    for m, model in enumerate(models):
        for j, d in enumerate(datasets):
            total_detections = []
            for i in range(n_iterations):
                model_dir = model + '_i0{}'.format(i)
                result_file = work_dir + model_dir + '/test_' + d + '/' + 'results_iou{}.pkl'.format(iou)
                if "snake" in model:
                    result_file = work_dir + model + '{}_boxes{}-{}_iou{}_i0{}.pkl'.format(d, 0, 2.0, iou, i)
                try:
                    results = load_file(result_file)
                    total_detections.append(sum_results(results['results']))
                except FileNotFoundError:
                    continue

            m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
            precision[j] = m_p
            recall[j] = m_r
            err_p[j] = std_p
        meanAp = np.mean(precision, 0)
        plt.plot(np.mean(recall, 0), np.mean(precision, 0), 'x--')
        results_on_real.append(np.round(np.mean(meanAp),2))  # , np.mean(np.mean(err_p, 0))

    frame['Real Data' + str(iou)] = pd.Series(results_on_real)
frame.set_index('Name')
plt.legend(names)
print(frame.to_string())
print(frame.to_latex())
plt.figure(figsize=(8, 3))

w = 1 / len(models)
w -= w * 0.1
plt.subplot(1, 2, 1)
plt.title('Simulated Data', fontsize=12)
for i, m in enumerate(models):
    plt.bar(np.arange(len(iou_thresh)) - w + i * w,
            [frame['Sim Data0.4'][i], frame['Sim Data0.6'][i], frame['Sim Data0.8'][i]], width=w)
    plt.xticks(np.arange(len(iou_thresh)), iou_thresh)
    plt.xlabel('Intersection Over Union')
    plt.ylabel('Average Precision')
    plt.ylim(0, 0.8)

plt.subplot(1, 2, 2)
plt.title('Real Data', fontsize=12)
for i, m in enumerate(models):
    plt.bar(np.arange(len(iou_thresh)) - w + i * w,
            [frame['Real Data0.4'][i], frame['Real Data0.6'][i], frame['Real Data0.8'][i]], width=w)
    plt.xticks(np.arange(len(iou_thresh)), iou_thresh)
    plt.xlabel('Intersection Over Union')
    plt.ylabel('Average Precision')
    plt.ylim(0, 0.8)

plt.legend(names)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/pp_bar.png')
plt.show(True)
