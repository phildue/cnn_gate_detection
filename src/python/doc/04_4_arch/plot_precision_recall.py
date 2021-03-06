import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.evaluation import interpolate
from utils.workdir import cd_work

cd_work()
models = [
    'racecourt',
    'width1',
    'width2',
    'width3',
    'width4',
]

dataset = 'iros2018_course_final_simple_17gates'

legend = [
    '$\\frac{1}{1}$',
    '$\\frac{1}{2}$',
    '$\\frac{1}{4}$',
    '$\\frac{1}{8}$',
    '$\\frac{1}{16}$',
]

n_iterations = 4

bins = None
n_objects = None
plt.figure(figsize=(8, 3))
plt.title('Precision - Recall of varying widths'.format(dataset))
w = 0.8 / len(models)

# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
for i_m, r in enumerate(models):
    precisions = []
    for it in range(n_iterations):
        try:
            frame = pd.read_pickle(
                'out/{0:s}_i{1:02d}/test_{2:s}/results_total_pr.pkl'.format(r, it, dataset))
            precision_raw = frame['{2:s}_p{0:02f}_i{1:02d}'.format(0.6, it, dataset)]
            recall_raw = frame['{2:s}_r{0:02f}_i{1:02d}'.format(0.6, it, dataset)]
            n_objects = frame['{} Objects'.format(dataset)]
            bins = frame['Sizes Bins']

            precision, _ = interpolate(precision_raw[0], recall_raw[0], np.linspace(0, 1.0, 11))

            precisions.append(precision)
        except (KeyError, FileNotFoundError) as e:
            print(e)
    precision = np.mean(precisions, 0)
    err = np.std(precisions, 0)

    plt.plot(np.linspace(0, 1.0, 11),precision,'x--')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.4, hspace=0.4)
plt.legend(legend, bbox_to_anchor=(1.0, 1.0),loc='upper right')
plt.ylim(0, 1.1)
plt.xlim(0, 1.1)
plt.savefig('doc/thesis/fig/precision_recall_width.png')

plt.show(True)
