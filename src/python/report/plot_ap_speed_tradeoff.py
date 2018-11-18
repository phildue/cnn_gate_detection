import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    # 'mavnet',
    'mavnet_lowres160',
    'mavnet_lowres320',
    'mavnet_strides',
    'mavnet_strides3_pool2',
    'mavnet_strides4_pool1',
    'yolov3_width0'
]
work_dir = 'out/'
n_iterations = 1

names = [
    # 'mavnet',
    'mavnet_160x120',
    'mavnet_320x240',
    'mavnet_320x240_strides',
    'mavnet_320x240_strides3_pool2',
    'mavnet_320x240_strides4_pool1',
    'tinyyolov3_160x120'
]

symbols = [
    ('o', 'r'),
    ('v', 'r'),
    ('v', 'g'),
    ('v', 'b'),
    ('v', 'c'),
    ('o', 'g'),
]
markers = ["o", "v", "^", "<", ">"]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
t = [
    50,
    100,
    40,
    60,
    50,
    700,
]
iou = 0.6
simset = 'iros2018_course_final_simple_17gates'
realsets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]
frame = pd.DataFrame()
frame['Name'] = pd.Series(names)
frame['Time'] = pd.Series(t)
results_on_sim = []
weights = []
for m, model in enumerate(models):
    total_detections = []

    for i in range(n_iterations):
        model_dir = model  # + '_i0{}'.format(i)
        result_file = work_dir + model_dir + '/test_' + simset + '/' + 'results_iou{}.pkl'.format(iou)
        try:
            results = load_file(result_file)
            total_detections.append(sum_results(results['results']))
        except FileNotFoundError:
            print("Not found: {}".format(model_dir))

    m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
    meanAp = np.mean(m_p)
    errAp = np.mean(std_p)
    results_on_sim.append(np.round(meanAp, 2))  # , errAp

    w = load_file(work_dir + model + '/summary.pkl')['weights']
    weights.append(w)

frame['Sim Data' + str(iou)] = pd.Series(results_on_sim)
frame['Weights'] = pd.Series(weights)

results_on_real = []
for m, model in enumerate(models):
    total_detections = []
    for i in range(n_iterations):
        detections_set = []
        for j, d in enumerate(realsets):
            model_dir = model# + '_i0{}'.format(i)
            result_file = work_dir + model_dir + '/test_' + d + '/' + 'results_iou{}.pkl'.format(iou)
            try:
                results = load_file(result_file)
                detections_set.append(sum_results(results['results']))
            except FileNotFoundError:
                print("Not found: {}".format(model_dir))
        if len(detections_set) > 0:
            total_detections.append(sum_results(detections_set))
    m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
    meanAp = np.mean(m_p, 0)
    errAP = np.mean(std_p, 0)
    results_on_real.append(np.round(meanAp, 2))  # , errAp

frame['Real Data' + str(iou)] = pd.Series(results_on_real)
frame.set_index('Name')

plt.figure(figsize=(10, 4))

w = 0.1 / len(models)
maxw = 1000000
plt.title('Speed Accuracy Trade-Off', fontsize=12)
plt.xlabel('Inference Time/Sample [ms]')
plt.ylabel('Average Precision')
plt.ylim(0, 1.1)

for i, m in enumerate(frame['Name']):
    plt.plot(frame['Time'][i], frame['Sim Data' + str(iou)][i], marker=symbols[i][0], color=symbols[i][1])
    plt.plot(frame['Time'][i], frame['Real Data' + str(iou)][i], marker=symbols[i][0], color=symbols[i][1])
# plt.plot(frame['Time'], frame['Real Data' + str(iou)], 'o')
plt.legend(frame['Name'], bbox_to_anchor=(1.0, 1.05))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
print(frame.to_string())
print(frame.to_latex())
plt.savefig('doc/thesis/fig/ap_speed_tradeoff.png')
plt.show(True)
