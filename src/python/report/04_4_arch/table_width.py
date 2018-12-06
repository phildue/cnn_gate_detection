import numpy as np
import pandas as pd

from utils.fileaccess.utils import save_file
from utils.workdir import cd_work

cd_work()
models = [
    'racecourt',
    'width3',
    'width4',

]
datasets = [
    'test_iros_gate',
    'iros2018_course_final_simple_17gates',
]

datasets_title = [
    'IROS',
    'Simulated MAV Race',
]

iou = 0.6
n_iterations = 2
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

column_names = ['1', '1/8', '1/16']
table = pd.DataFrame()
table['Width'] = column_names

for i_d, d in enumerate(datasets):
    column_content = []
    for i_m, _ in enumerate(models):
        results = []
        for i_i in range(n_iterations):
            result = frame['{}_i0{}'.format(d, i_i)][i_m]
            if result >= 0:
                results.append(result)
        column_content.append('{:.2f} +- {:.2f}'.format(np.mean(results), np.std(results)))
    table[datasets_title[i_d]] = column_content

print(table.to_string(index=False))
print(table.to_latex(index=False))
save_file(table.to_latex(index=False), 'depth.txt', 'doc/thesis/tables/', raw=True)
