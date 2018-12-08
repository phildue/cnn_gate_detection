import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.evaluation import interpolate
from utils.workdir import cd_work

cd_work()
models = [
    'ewfo_sim',
    'randomview',
    'racecourt',
    # 'racecourt_allviews',
    'randomview_and_racecourt_allviews',
]

dataset = 'iros2018_course_final_simple_17gates'

legend = [
    'Frontal Views',
    'Random Placement',
    'Simulated Flight',
    # 'Simulated Flight - All View Points',
    'Combined'
]

n_iterations = 2

bins = None
n_objects = None
plt.figure(figsize=(8, 3))
plt.title('Precision/Recall for different Object Sizes on Simulated MAV Race'.format(dataset))
w = 0.8 / len(models)
colors = [(0, 0, 0.8), (0, 0.8, 0.8), (0, 0.8, 0), (0.8, 0, 0)]
colors_dark = [(0, 0, 0.5), (0, 0.5, 0.5), (0, 0.5, 0), (0.5, 0, 0)]

# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
handles = []
for i_m, r in enumerate(models):
    precisions = []
    recalls = []
    for it in range(n_iterations):
        try:
            frame = pd.read_pickle(
                'out/{0:s}_i{1:02d}/test_{2:s}/results_size_pr.pkl'.format(r, it, dataset))
            n_objects = frame['{} Objects'.format(dataset)]
            bins = frame['Sizes Bins']

            precisions_bin = []
            recalls_bin = []
            precision_raw = frame['{2:s}_p{0:02f}_i{1:02d}'.format(0.6, it, dataset)]
            recall_raw = frame['{2:s}_r{0:02f}_i{1:02d}'.format(0.6, it, dataset)]
            for i_b in range(len(bins)):
                precision, recall = interpolate(precision_raw[i_b], recall_raw[i_b], np.linspace(0, 1.0, 11))
                precisions_bin.append(np.array(precision_raw[i_b]))
                recalls_bin.append(np.array(recall_raw[i_b]))
            precisions.append(precisions_bin)
            recalls.append(recalls_bin)


        except (KeyError, FileNotFoundError) as e:
            print(e)

    precision = np.mean(precisions, 0)
    err = np.std(precisions, 0)
    recall = np.mean(recalls, 0)
    h = plt.bar(np.arange(len(bins)) + i_m * w - len(models) * w, [p[4] for p in precision], width=w, zorder=2,
                color=colors[i_m])
    plt.bar(np.arange(len(bins)) + i_m * w - len(models) * w, [r[4] for r in recall], width=w, zorder=2,
            color=colors_dark[i_m])

    # plt.errorbar(np.arange(len(bins)) + i_m * w - (len(models)) * w, precision[4], err[4], 0, fmt=' ', ecolor='gray', capsize=2,
    #              elinewidth=1, zorder=3)

    handles += [h]

    # recall[0] = 1.0
    # plt.errorbar(recall[:-1], precision[:-1], err[:-1], 0, 'x--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(legend, bbox_to_anchor=(0.6, 0.4))
plt.ylim(0, 1.1)
plt.xlim(0, 1.1)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.4, hspace=0.4)

# plt.ylim(0, 0.8)
plt.xlabel('Label bins as fraction of image size', horizontalalignment='right', x=1.0)
plt.ylabel('$p_{60}^{0.5}$/$r_{60}^{0.5}$')
plt.xticks(np.arange(len(bins) + 1) - 1,
           ['$\\frac{1}{100}$', '$\\frac{1}{32}$', '$\\frac{1}{16}$', '$\\frac{1}{8}$', '$\\frac{1}{4}$',
            '$\\frac{1}{2}$', '1.0'], fontsize=12)
plt.legend(handles, legend, bbox_to_anchor=(1.0, 1.0), loc='upper left')
# Shrink current axis by 20%
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None)
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.savefig('doc/thesis/fig/view_precision_recall.png')

plt.show(True)
plt.show(True)
