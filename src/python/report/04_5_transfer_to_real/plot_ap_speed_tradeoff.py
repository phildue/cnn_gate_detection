import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.evaluation import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    'blur_distortion',
    '320_strides1',
    '320_strides2',
    '160',
    # 'mavnet_strides',
    # 'mavnet_strides3_pool2',
    # 'mavnet_strides4_pool1',
    'yolo_lowres160'
]
work_dir = 'out/'

names = models

symbols = [
    ('>', 'r'),
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
    900,
    500,
    200,
    100,
    # 40,
    # 60,
    # 50,
    700,
    20,
]
datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    # 'jevois_hallway',
]



ious = [0.6]
n_iterations = 5
frame = pd.DataFrame()
frame['Name'] = models + ['SnakeGate']
frame['Time'] = t
for iou in ious:
    for d in datasets:
        for it in range(n_iterations):
            column = []
            for m in models:
                try:
                    new_frame = pd.read_pickle('out/{0:s}_i{1:02d}/test_{2:s}/results_total.pkl'.format(m, it, d))
                    cell = new_frame['{}_ap{:.6f}_i0{}'.format(d, iou, it)][0]
                    column.append(cell)
                except (FileNotFoundError, KeyError) as e:
                    column.append(-1)
                    print(e)
                    continue

            result_file = 'out/snake/test_' + d + '_results_iou' + str(iou) + '_' + str(it) + '.pkl'.format(d,
                                                                                                           iou,
                                                                                                           it)

            results = load_file(result_file)
            mean_pr, mean_rec, std_pr, std_rec = average_precision_recall([sum_results(results['results'])])
            column.append(mean_pr.mean())

            frame['{}_iou{}_i0{}'.format(d, iou, it)] = column
            print(frame.to_string())

frame.set_index('Name')

plt.figure(figsize=(10, 4))
iou=0.6
w = 0.1 / len(models)
maxw = 1000000
plt.title('Speed Accuracy Trade-Off', fontsize=12)
plt.xlabel('Inference Time/Image [ms]')
plt.ylabel('$ap_{60}$')
plt.ylim(0, 0.6)

handles = []
for i, m in enumerate(frame['Name']):
    aps_datasets = []
    errs_datasets = []
    for d in datasets:
        aps = []
        for it in range(n_iterations):
            ap = frame['{}_iou{}_i0{}'.format(d, iou, it)][i]
            if ap >= 0:
                aps.append(ap)
        aps_datasets.append(np.mean(aps))
        errs_datasets.append(np.std(aps))

    h = plt.errorbar(frame['Time'][i], np.mean(aps_datasets), yerr=np.mean(errs_datasets),
                     marker=symbols[i][0], color=symbols[i][1], elinewidth=1, capsize=2)
    handles.append(h[0])
    # plt.plot(frame['Time'][i], frame['Real Data' + str(iou)][i], marker=symbols[i][0], color=symbols[i][1])
# plt.plot(frame['Time'], frame['Real Data' + str(iou)], 'o')
plt.legend(handles, frame['Name'], bbox_to_anchor=(1.0, 1.05))
plt.grid(b=True, which='major', color=(0.75, 0.75, 0.75), linestyle='-')
plt.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--')
plt.minorticks_on()

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
print(frame.to_string())
print(frame.to_latex())
plt.savefig('doc/thesis/fig/ap_speed_tradeoff.png')


# plt.title('Speed Accuracy Trade-Off Real Data', fontsize=12)
# plt.xlabel('Inference Time/Sample [ms]')
# plt.ylabel('Average Precision')
# plt.ylim(0, 1.1)
#
# handles = []
# for i, m in enumerate(frame['Name']):
#     h = plt.errorbar(frame['Time'][i], frame['Real Data' + str(iou)][i], yerr=frame['Real Data Err' + str(iou)][i],
#                      marker=symbols[i][0], color=symbols[i][1], elinewidth=1, capsize=2)
#     handles.append(h[0])
#     # plt.plot(frame['Time'][i], frame['Real Data' + str(iou)][i], marker=symbols[i][0], color=symbols[i][1])
# # plt.plot(frame['Time'], frame['Real Data' + str(iou)], 'o')
# plt.legend(handles, frame['Name'], bbox_to_anchor=(1.0, 1.05))
# plt.grid(b=True, which='major', color=(0.75, 0.75, 0.75), linestyle='-')
# plt.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--')
# plt.minorticks_on()
#
# plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
#                     wspace=0.3, hspace=0.3)
# print(frame.to_string())
# print(frame.to_latex())
# # plt.savefig('doc/thesis/fig/ap_speed_tradeoff_real.png')
plt.show(True)
