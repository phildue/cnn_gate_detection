import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
plt.figure(figsize=(9, 3))
plt.title('Tested on Synthetic Test Set'.format(dataset))
w = 0.8 / len(models)

# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
for i_m, r in enumerate(models):
    aps = []
    for it in range(n_iterations):
        try:
            frame = pd.read_pickle(
                'out/{0:s}_i{1:02d}/test_{2:s}/results_size_cluster.pkl'.format(r, it, dataset))
            ap = frame['{2:s}_ap{0:02f}_i{1:02d}'.format(0.6, it, dataset)]
            n_objects = frame['{} Objects'.format(dataset)]
            bins = frame['Sizes Bins']

            aps.append(ap)
        except (KeyError, FileNotFoundError) as e:
            print(e)
    ap = np.mean(aps, 0)
    err = np.std(aps, 0)

    plt.bar(np.arange(len(bins)) + i_m * w - len(models) * w, ap, width=w, yerr=err)

plt.text(-1.15, 1.0, '$N_{Objects}$:',color='gray')
for x, y in enumerate(n_objects):
    plt.text(x - 0.5, 1.0, '${}$'.format(np.round(y, 2)), color='gray', fontweight='bold')
plt.ylim(0, 1.1)
plt.xlabel('$A_O/A_I$')
plt.ylabel('$ap_{60}$')
plt.xticks(np.arange(len(bins))-1, np.round(bins, 3))
plt.legend(legend, bbox_to_anchor=(1.0, 1.0),loc='upper left')
# Shrink current axis by 20%
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None)
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.savefig('doc/thesis/fig/view_size.png')

plt.show(True)
