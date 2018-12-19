import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()
models = [
    'ewfo',
    'ewfo_voc',
    'ewfo_sim',
]

datasets = [
    'test_iros_gate',
    # 'test_iros_cats',
    # 'test_iros_sign',
]

xlabel = [
    'Single Environment from Simulator',
    'Background from Dataset',
    'Multiple Environments from Simulator',
]
legend = [
    'Testset Gate',
    # 'Testset Gate with Cats',
    # 'Testset Gate with Sign',
]

title = ['gate $[ap_{60}]$', 'sign $[ap_{60}]$', 'cats $[ap_{60}]$']
iou = 0.6
n_iterations = 4
frame = pd.DataFrame()
frame['Name'] = models
for d in datasets:
    for it in range(n_iterations):
        column = []
        for m in models:
            try:
                new_frame = pd.read_pickle('out/{0:s}_i{1:02d}/test_{2:s}/results_total.pkl'.format(m, it, d))
                cell = new_frame['{}_ap{:.6f}_i0{}'.format(d, iou, it)][0]
                column.append(cell)
            except FileNotFoundError as e:
                column.append(-1)
                print(e)
                continue
        frame['{}_i0{}'.format(d, it)] = column
        print(frame.to_string())

setname = 'iros'
columns = ['gate']  # , 'sign', 'cats']

plt.figure(figsize=(9, 3))
w = 0.8 / len(models)
plt.title("Test in Unseen Environment")
plt.grid(b=True, which='major', color=(0.75, 0.75, 0.75), linestyle='-', zorder=0)
plt.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--', zorder=0)
handles = []
for i_c, c in enumerate(columns):
    aps = []
    errs = []

    for i_m, m in enumerate(models):
        results = []
        for i_i in range(n_iterations):
            result = frame['test_{}_{}_i0{}'.format(setname, c, i_i)][i_m]
            if result >= 0:
                results.append(result)
        aps.append(np.mean(results))
        errs.append(np.std(results))

    h = plt.bar(np.arange(len(models)) + i_c * w - 0.5 * len(columns) * w + 0.5 * w, aps, width=w, zorder=2)
    handles += [h]
    # plt.errorbar(np.arange(len(columns)) + i_c * w - 0.5 * len(columns) * w + 0.5 * w, aps, errs, 0, fmt=' ',
    #              ecolor='gray', capsize=2,
    #              elinewidth=1, zorder=3)
plt.xticks(np.arange(len(models)), xlabel)
# plt.xlabel('Training Method', horizontalalignment='left', x=0.0)
plt.ylabel('Average Precision')
plt.minorticks_on()
# plt.legend(handles, legend, bbox_to_anchor=(0.0, 1.0), loc='upper left')
# Shrink current axis by 20%
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None)
plt.savefig('doc/presentation/background.png', dpi=600)
plt.show()
