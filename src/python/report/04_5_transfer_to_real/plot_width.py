import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluation.utils import average_precision_recall, sum_results

from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    'yolov3_width0',
    'yolov3_width1',
    'yolov3_width2',
    'yolov3_width3',
    # 'yolo_width4',
]
work_dir = 'out/'
n_iterations = 1

names = models
width = [
    1,
    0.5,
    0.25,
    0.125,
    # 0.0625
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
for m, model in enumerate(models):
    total_detections = []

    for i in range(n_iterations):
        model_dir = model #+ '_i0{}'.format(i)
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

frame.set_index('Name')

plt.figure(figsize=(8, 3))

w = 0.1 / len(models)
maxw = 1000000
plt.title('Performance Across Width', fontsize=12)
plt.bar(1 / np.array(width), frame['Sim Data' + str(iou)], width=w)

# plt.bar(frame['Weights']/maxw, frame['Real Data' + str(iou)],width=w)
plt.xlabel('Width Multiplier')
plt.ylabel('Average Precision')
plt.ylim(0, 1.1)
# plt.legend(['Sim Data', 'Real Data'], bbox_to_anchor=(1.1, 1.05))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
print(frame.to_string())
print(frame.to_latex())
plt.savefig('doc/thesis/fig/perf_width.png')
plt.show(True)
