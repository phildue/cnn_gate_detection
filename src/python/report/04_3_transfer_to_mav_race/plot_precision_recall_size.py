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
    'racecourt_allviews',
    'randomview_and_racecourt_allviews',
]

dataset = 'iros2018_course_final_simple_17gates'

legend = [
    'Frontal Views',
    'Random Placement',
    'Simulated Flight',
    'Simulated Flight - All View Points',
    'Combined'
]

n_iterations = 1

bins = None
n_objects = None
plt.figure(figsize=(8, 3))
plt.title('Tested on {}'.format(dataset))
w = 0.8 / len(models)

# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
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
                precision, _ = interpolate(precision_raw[i_b], recall_raw[i_b], np.linspace(0, 1.0, 11))
                precisions_bin.append(np.array(precision_raw[i_b]))
                recalls_bin.append(np.array(recall_raw[i_b]))
            precisions.append(precisions_bin)
            recalls.append(recalls_bin)


        except (KeyError, FileNotFoundError) as e:
            print(e)

    for i_b in range(len(bins)):
        plt.subplot(1, 3, i_b + 1)
        plt.title('Bin{}'.format(bins[i_b]))
        precision = np.mean([p[i_b] for p in precisions], 0)
        err = np.std([p[i_b] for p in precisions], 0)
        recall = np.mean([r[i_b] for r in recalls], 0)
        # recall[0] = 1.0
        plt.errorbar(recall[:-1], precision[:-1], err[:-1], 0, 'x--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(legend, bbox_to_anchor=(0.6, 0.4))
        plt.ylim(0, 1.1)
        plt.xlim(0, 1.1)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.4, hspace=0.4)

# plt.savefig('doc/thesis/fig/view_size.png')

plt.show(True)
