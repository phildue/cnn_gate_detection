import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()
models = [
    'racecourt',
    'width1',
    'width2',
    'width3',
    'width4',

]
datasets = [
    # 'test_basement_gate',
    'iros2018_course_final_simple_17gates',
]

datasets_title = [
    # 'Simple Test Set',
    'Simulated MAV Race',
]

ious = [0.6]
n_iterations = 4
frame = pd.DataFrame()
frame['Name'] = models
for iou in ious:
    for d in datasets:
        for it in range(n_iterations):
            column = []
            weights = []
            for m in models:
                try:
                    new_frame = pd.read_pickle('out/{0:s}_i{1:02d}/test_{2:s}/results_total.pkl'.format(m, it, d))
                    # w = ModelSummary.from_file('out/{0:s}_i{1:02d}/model.h5'.format(m, it)).weights
                    cell = new_frame['{}_ap{:.6f}_i0{}'.format(d, iou, it)][0]
                    column.append(cell)
                    # weights.append(w)
                except FileNotFoundError as e:
                    column.append(-1)
                    print(e)
                    continue
            frame['{}_iou{}_i0{}'.format(d, iou, it)] = column
            print(frame.to_string())

column_names = ['$1$', '$\\frac{1}{2}$', '$\\frac{1}{4}$', '$\\frac{1}{8}$', '$\\frac{1}{16}$']
table = pd.DataFrame()
table['Width'] = column_names
iou = 0.6
for i_d, d in enumerate(datasets):
    column_content = []
    for i_m, _ in enumerate(models):
        results = []
        for i_i in range(n_iterations):
            result = frame['{}_iou{}_i0{}'.format(d, iou, i_i)][i_m]
            if result >= 0:
                results.append(result)
        column_content.append(np.mean(results))
    table['ap_60'] = column_content

table['Weights'] = [12133646, 3039638, 763034, 192332, 48881]
table['Reduction in Weights [\%]'] = np.round((np.array([12133646, 3039638, 763034, 192332, 48881]) / 12133646), 1)

plt.figure(figsize=(9, 3))
w = 0.8 / len(models)
plt.grid(b=True, which='major', color=(0.75, 0.75, 0.75), linestyle='-', zorder=0)
plt.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--', zorder=0)
# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')

plt.plot(table['Width'], table['ap_60'], 'x--')
plt.plot(table['Width'], table['Reduction in Weights [\%]'], 'x--')
# plt.ylabel('$ap_{60}$')
plt.xlabel('Width')
plt.xticks(column_names)
plt.ylim(-0.1, 0.6)
plt.legend(['Average Precision', 'Network Size [% in Weights]'], bbox_to_anchor=(0.0, 0.0), loc='lower left')
# Shrink current axis by 20%
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None)
# ax = plt.subplot(111)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.savefig('doc/presentation/width.png',dpi=600)

plt.show(True)
