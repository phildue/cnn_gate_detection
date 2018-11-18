import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    'datagen/yolov3_d0_416x416',
    'objectdetect/yolov3_w0_416x416',
    'datagen/yolov3_d1_416x416',
    'datagen/yolov3_d2_416x416',
    # 'objectdetect/yolov3_w3_416x416',
    # 'datagen/yolov3_blur416x416',
    # 'datagen/yolov3_arch2416x416',

]

work_dir = 'out/thesis/'
n_iterations = 2

names = [
    'd0',
    'w0',
    'd1',
    'd2',
    'd01'
    # 'arch'
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

results_on_sim = []
weights = []
layers = []
for m, model in enumerate(models):
    total_detections = []

    for i in range(n_iterations):
        model_dir = model + '_i0{}'.format(i)
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
    summary = load_file(work_dir + model + '_i00/summary.pkl')
    w = summary['weights']
    weights.append(w)
    l = 0
    for layer in summary['architecture']:
        if 'conv' in layer['name']:
            l += 1
    layers.append(l)

results_on_sim.append(0.39)
weights.append(51.994)
layers.append(7)

frame['Sim Data' + str(iou)] = pd.Series(results_on_sim)
frame['Weights'] = pd.Series(weights)
frame['Layers'] = pd.Series(layers)

results_on_real = []
for m, model in enumerate(models):
    total_detections = []
    for i in range(n_iterations):
        detections_set = []
        for j, d in enumerate(realsets):
            model_dir = model + '_i0{}'.format(i)
            result_file = work_dir + model_dir + '/test_' + d + '/' + 'results_iou{}.pkl'.format(iou)
            try:
                results = load_file(result_file)
                detections_set.append(sum_results(results['results']))
            except FileNotFoundError:
                continue
        if len(detections_set) > 0:
            total_detections.append(sum_results(detections_set))
    m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
    meanAp = np.mean(m_p, 0)
    errAP = np.mean(std_p, 0)
    results_on_real.append(np.round(meanAp, 2))  # , errAp

results_on_real.append(0.39)
frame['Real Data' + str(iou)] = pd.Series(results_on_real)
frame.set_index('Name')

plt.figure(figsize=(8, 3))

w = 1 / len(models)
plt.title('Performance Across Depth', fontsize=12)
plt.bar(frame['Layers'], frame['Sim Data' + str(iou)], width=w)

# plt.bar(frame['Layers'], frame['Real Data' + str(iou)], width=w)
plt.xlabel('Layers')
plt.ylabel('Average Precision')
plt.ylim(0, 1.1)
# plt.legend(['Sim Data', 'Real Data'], bbox_to_anchor=(1.1, 1.05))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.3, hspace=0.3)
print(frame.to_string())
print(frame.to_latex())
plt.savefig('doc/thesis/fig/perf_depth.png')
plt.show(True)
