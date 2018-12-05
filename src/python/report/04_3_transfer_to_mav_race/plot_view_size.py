import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

n_iterations = 2


bins = None
n_objects = None
plt.figure(figsize=(8, 3))
plt.title('Tested on {}'.format(dataset))
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

    plt.bar(np.arange(len(bins)) + i_m * w - len(bins)*0.5*w, ap, width=w, yerr=err)

for x, y in enumerate(n_objects):
    plt.text(x - 0.1, 1.0, str(np.round(y, 2)), color='gray', fontweight='bold')
plt.xlabel('Area Relative to Image Size')
plt.ylabel('Average Precision')
plt.xticks(np.arange(len(bins)), np.round(bins, 3))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.4, hspace=0.4)
plt.legend(legend,bbox_to_anchor=(0.6,0.4))
plt.ylim(0, 1.1)
plt.savefig('doc/thesis/fig/view_size.png')

plt.show(True)
