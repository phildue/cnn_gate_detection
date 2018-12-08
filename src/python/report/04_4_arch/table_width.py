import numpy as np
import pandas as pd

from utils.fileaccess.utils import save_file
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
            frame['{}_iou{}_i0{}'.format(d, iou,it)] = column
            print(frame.to_string())

column_names = ['$1$', '$\\frac{1}{2}$','$\\frac{1}{4}$', '$\\frac{1}{8}$', '$\\frac{1}{16}$']
table = pd.DataFrame()
table['Width'] = column_names
iou=0.6
for i_d, d in enumerate(datasets):
    column_content = []
    for i_m, _ in enumerate(models):
        results = []
        for i_i in range(n_iterations):
            result = frame['{}_iou{}_i0{}'.format(d, iou,i_i)][i_m]
            if result >= 0:
                results.append(result)
        column_content.append('${:.2f} \pm {:.2f}$'.format(np.mean(results), np.std(results)))
    table[datasets_title[i_d] + ' $ap_{' + str(int(np.round(iou*100,0))) + '}$'] = column_content

print(table.to_string(index=False))
print(table.to_latex(index=False,escape=False))
save_file(table.to_latex(index=False,escape=False), 'width.txt', 'doc/thesis/tables/', raw=True)
