import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.fileaccess.utils import save_file
from utils.workdir import cd_work

cd_work()
models = [
    'mavlabgates',
    'distortion',
    'blur',
    'hsv',
    'exposure',
    'chromatic',

]
titles = [
    'no augmentation',
    'distortion',
    'blur',
    'hsv',
    'exposure',
    'chromatic',

]

datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]

datasets_title = [
    'Cyberzoo',
    'Basement',
    'Hallway'
]

ious = [0.4, 0.6]
n_iterations = 2
frame = pd.DataFrame()
frame['Name'] = models
for iou in ious:
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
            frame['{}_iou{}_i0{}'.format(d, iou, it)] = column
            print(frame.to_string())

column_names = titles
table = pd.DataFrame()
table['Augmentation'] = column_names
iou = 0.6
for i_d, d in enumerate(datasets):
    column_content = []
    for i_m, _ in enumerate(models):
        results = []
        for i_i in range(n_iterations):
            result = frame['{}_iou{}_i0{}'.format(d, iou, i_i)][i_m]
            if result >= 0:
                results.append(result)
        column_content.append('${:.2f} \pm {:.2f}$'.format(np.mean(results), np.std(results)))
    table[datasets_title[i_d] + ' $ap_{' + str(int(np.round(iou * 100, 0))) + '}$'] = column_content

print(table.to_string(index=False))
print(table.to_latex(index=False, escape=False))
save_file(table.to_latex(index=False, escape=False), 'augmentation.txt', 'doc/thesis/tables/', raw=True)

plt.figure(figsize=(8, 3))
w = 1 / len(models)
handles = []
for i_d, d in enumerate(datasets):
    mean = []
    err = []
    for i_m, m in enumerate(models):
        results = []
        for i_i in range(n_iterations):
            result = frame['{}_iou{}_i0{}'.format(d, iou, i_i)][i_m]
            if result >= 0:
                results.append(result)
        mean.append(np.mean(results))
        err.append(np.std(results))
    h = plt.bar(np.arange(len(models)) + i_d * w - len(models) * w, mean, width=w, capsize=2, ecolor='gray')
    plt.errorbar(np.arange(len(models)) + i_d * w - len(models) * w, mean, err, 0, fmt=' ', ecolor='gray', capsize=2,
                 elinewidth=1, )
    handles.append(h)

plt.xticks(np.arange(len(models)) - 1, titles)
plt.xlabel('Augmentation Method')
plt.ylabel('$ap_{60}$')
plt.ylim(0, 0.6)
plt.legend(handles, datasets_title)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/augmentation.png')
plt.show(True)
