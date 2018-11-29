import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()
models = [
    'sign',
    'cats',
    'ewfo',
]

datasets = [
    'test_basement_sign',
    'test_basement_cats',
    'test_basement_gate',
]

datasets_transfer = [
    'test_iros_sign',
    'test_iros_cats',
    'test_iros_gate',
]

legend = [
    'Sign',
    'Cats',
    'Racing Gate',
]

n_iterations = 2


plt.figure(figsize=(8, 3))
plt.title('Tested in Dark')
w = 0.1
bins = None
n_objects = None
for i_m, r in enumerate(models):

    drop = []
    for it in range(n_iterations):
        try:
            frame = pd.read_pickle('out/{0:s}_i{1:02d}/test_{2:s}/results_size_cluster.pkl'.format(r, it, datasets[i_m]))
            ap = frame['{2:s}_ap{0:02f}_i{1:02d}'.format(0.6, it, datasets[i_m])]
            frame_transfer = pd.read_pickle('out/{0:s}_i{1:02d}/test_{2:s}/results_size_cluster.pkl'.format(r, it, datasets_transfer[i_m]))
            ap_t = frame_transfer['{2:s}_ap{0:02f}_i{1:02d}'.format(0.6, it, datasets_transfer[i_m])]
            drop.append(ap_t)
            bins = frame['Sizes Bins']
            n_objects = frame['{} Objects'.format(datasets[i_m])]
        except KeyError as e:
            print(e)
    mean_drop = np.mean(drop, 0)
    err = np.std(drop, 0)

    plt.bar(np.arange(len(bins)) + i_m * w - w * (len(models)-1) / 2, mean_drop, width=w, yerr=err)

for x, y in enumerate(n_objects):
    plt.text(x - 0.1, 1.0, str(np.round(y, 2)), color='gray', fontweight='bold')
plt.xlabel('Area Relative to Image Size')
plt.ylabel('Average Precision')
plt.xticks(np.arange(len(bins)-1), np.round(bins[1:], 3))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.4, hspace=0.4)
plt.legend(legend, loc='upper left')
plt.ylim(0, 1.1)
plt.savefig('doc/thesis/fig/ewfo_vs_other_transfer_size.png')

plt.show(True)
