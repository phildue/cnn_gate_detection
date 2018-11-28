import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()
models = [
    'ewfo',
    'sign',
    'cats',
    'ewfo_deep',
    'sign_deep',
    'cats_deep',
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

n_iterations = 1
frames = []
for m in models:
    frame = None
    for d in datasets:
        for it in range(n_iterations):
            try:
                new_frame = pd.read_pickle('out/{0:s}_i{1:02d}/test_{2:s}/results_size_cluster.pkl'.format(m, it, d))
                # if frame is None:
                frame = new_frame
                # else:
                #     frame.merge(new_frame)

            except FileNotFoundError as e:
                print(e)
                continue
    if frame is not None:
        print(frame.to_string())
        frames.append(frame)

bins = frames[0]['Sizes Bins']

table_shallow = pd.DataFrame()

column_names = ['EWFO', 'Sign', 'Cats']
columns = ['gate', 'sign', 'cats']


table_shallow['Trained'] = column_names

for c in columns:
    column_content = []
    for i_m, m in enumerate(columns):
        result = frames[i_m]['test_basement_' + c]
        column_content.append(result)
    table_shallow['Basement ' + c] = column_content

for c in columns:
    column_content = []
    for i_m, m in enumerate(columns):
        result = frames[i_m]['test_iros_' + c]
        column_content.append(result)
    table_shallow['IROS ' + c] = column_content

print(table_shallow.to_string())

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
