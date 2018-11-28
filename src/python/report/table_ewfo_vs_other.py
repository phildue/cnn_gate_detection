import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.fileaccess.utils import save_file
from utils.workdir import cd_work

cd_work()
models = [
    'ewfo',
    'sign',
    'cats',
    # 'ewfo_deep',
    # 'sign_deep',
    # 'cats_deep',
]

datasets = [
    'test_iros_gate',
    'test_iros_sign',
    'test_iros_cats',
    'test_basement_gate',
    'test_basement_sign',
    'test_basement_cats',
]

legend = [
    'Trained on Sign',
    'Trained on Cats',
    'Trained on EWFO',
    'Trained on Sign Deep',
    'Trained on Cats Deep',
    'Trained on EWFO Deep',
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

table_shallow_basement = pd.DataFrame()

column_names = ['EWFO', 'Sign', 'Cats']
columns = ['gate', 'sign', 'cats']

table_shallow_basement['Trained/Tested'] = column_names

for i_c, c in enumerate(columns):
    column_content = []
    for i_m, m in enumerate(columns):
        result_mean = 0
        for i_i in range(n_iterations):
            result = frame['test_basement_{}_i0{}'.format(c, i_i)][i_m]
            result_mean += result / n_iterations
        column_content.append(result_mean)
    table_shallow_basement[column_names[i_c]] = column_content

print(table_shallow_basement.to_string(index=False))
print(table_shallow_basement.to_latex(index=False))

save_file(table_shallow_basement.to_latex(index=False), 'shallow_basement.txt', 'doc/thesis/tables/', raw=True)

table_shallow_iros = pd.DataFrame()
table_shallow_iros['Trained/Tested'] = column_names

for i_c, c in enumerate(columns):
    column_content = []
    for i_m, m in enumerate(columns):
        result_mean = 0
        for i_i in range(n_iterations):
            result = frame['test_iros_{}_i0{}'.format(c, i_i)][i_m]
            result_mean += result / n_iterations
        column_content.append(result_mean)
    table_shallow_iros[column_names[i_c]] = column_content

print(table_shallow_iros.to_string(index=False))
print(table_shallow_iros.to_latex(index=False))
save_file(table_shallow_iros.to_latex(index=False), 'shallow_iros.txt', 'doc/thesis/tables/', raw=True)

"""
Deep
"""
# table_deep = pd.DataFrame()
#
# columns = ['EWFO', 'Sign', 'Cats']
#
# table_shallow['Trained'] = columns
#
# for c in columns:
#     column_content = []
#     for i_m, m in enumerate(columns):
#         result = frames[i_m + 3]['test_basement_' + c]
#         column_content.append(result)
#     table_deep['Basement ' + c] = column_content
#
# for c in columns:
#     column_content = []
#     for i_m, m in enumerate(columns):
#         result = frames[i_m + 3]['test_iros_' + c]
#         column_content.append(result)
#     table_deep['IROS ' + c] = column_content
#
# print(table_shallow.to_string())
