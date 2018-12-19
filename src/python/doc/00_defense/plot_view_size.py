import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()
models = [
    # 'ewfo_sim',
    'randomview',
    'racecourt',
    # 'racecourt_allviews',
    # 'randomview_and_racecourt_allviews',
]

dataset = 'iros2018_course_final_simple_17gates'

legend = [
    # 'Frontal Views',
    'Random\n Placement',
    'Simulated\n Flight',
    # 'Simulated Flight \n- All View Points',
    # 'Combined'
]

n_iterations = 2

bins = None
n_objects = None
plt.figure(figsize=(9, 3))
plt.title('Test on Simulated MAV Race'.format(dataset))
w = 0.8 / len(models)
plt.grid(b=True, which='major', color=(0.75, 0.75, 0.75), linestyle='-',zorder=0)
plt.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--',zorder=0)
# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
handles = []
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

    h = plt.bar(np.arange(len(bins)) + i_m * w - len(models) * w, ap, width=w,zorder=2)
    plt.errorbar(np.arange(len(bins)) + i_m * w - (len(models)) * w, ap, err, 0, fmt=' ', ecolor='gray', capsize=2,
                 elinewidth=1, zorder=3)
    handles.append(h)
plt.text(-1.35, 0.7, '$N_{Objects}$:', color='gray',fontweight='bold')
for x, y in enumerate(n_objects):
    plt.text(x-0.7, 0.7, '${}$'.format(np.round(y, 2)), color='gray', fontweight='bold')
plt.ylim(0, 0.8)
plt.xlabel('Label bins as fraction of image size', horizontalalignment='right',x=1.0)
plt.ylabel('Average Precision')
plt.xticks(np.arange(len(bins)) - 1,
           ['$\\frac{1}{1000}$', '$\\frac{1}{32}$', '$\\frac{1}{16}$', '$\\frac{1}{8}$', '$\\frac{1}{4}$', '$\\frac{1}{2}$', '1.0'],fontsize=12)
plt.legend(handles,legend, bbox_to_anchor=(1.0, 1.0), loc='upper left')
# Shrink current axis by 20%
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None)
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
plt.minorticks_on()
plt.savefig('doc/presentation/view.png',dpi=600)

plt.show(True)
