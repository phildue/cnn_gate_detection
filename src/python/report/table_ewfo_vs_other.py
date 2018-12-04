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


def create_table(setname, filename, shift=0):
    column_names = ['Gate', 'Sign', 'Cats']
    columns = ['gate', 'sign', 'cats']
    table = pd.DataFrame()
    table['Trained/Tested'] = column_names

    for i_c, c in enumerate(columns):
        column_content = []
        for i_m, _ in enumerate(columns):
            results = []
            for i_i in range(n_iterations):
                result = frame['test_{}_{}_i0{}'.format(setname, c, i_i)][i_m + shift]
                if result >= 0:
                    results.append(result)
            column_content.append('{:.2f} +- {:.2f}'.format(np.mean(results), np.std(results)))
        table[column_names[i_c]] = column_content

    print(table.to_string(index=False))
    print(table.to_latex(index=False))
    save_file(table.to_latex(index=False), filename, 'doc/thesis/tables/', raw=True)


def create_table_all(setname, filename, shift=0):
    column_names = ['Gate', 'Sign', 'Cats', 'EWFO Deep', 'Sign Deep', 'Cats Deep']
    columns = ['gate', 'sign', 'cats', 'gate', 'sign', 'cats']
    table = pd.DataFrame()
    table['Trained/Tested'] = column_names

    for i_c, c in enumerate(columns[:3]):
        column_content = []
        for i_m, _ in enumerate(columns):
            results = []
            for i_i in range(n_iterations):
                result = frame['test_{}_{}_i0{}'.format(setname, c, i_i)][i_m]
                if result >= 0:
                    results.append(result)
            column_content.append('{:.2f} +- {:.2f}'.format(np.mean(results), np.std(results)))
        table[column_names[i_c]] = column_content

    print(table.to_string(index=False))
    print(table.to_latex(index=False))
    save_file(table.to_latex(index=False), filename, 'doc/thesis/tables/', raw=True)


def create_table_diff(filename):
    column_names = ['Gate', 'Sign', 'Cats', 'Gate Deep', 'Sign Deep', 'Cats Deep']
    columns = ['gate', 'sign', 'cats', 'gate', 'sign', 'cats']
    table = pd.DataFrame()
    table['Trained/Tested'] = column_names

    for i_c, c in enumerate(columns[:3]):
        column_content = []
        for i_m, _ in enumerate(columns):
            results = []
            for i_i in range(n_iterations):
                result = frame['test_iros_{}_i0{}'.format(c, i_i)][i_m]
                result_shift = frame['test_basement_{}_i0{}'.format(c, i_i)][i_m]
                if result >= 0 and result_shift >= 0:
                    results.append(result-result_shift)
            column_content.append('{:.2f} +- {:.2f}'.format(np.mean(results), np.std(results)))
        table[column_names[i_c]] = column_content

    print(table.to_string(index=False))
    print(table.to_latex(index=False))
    save_file(table.to_latex(index=False), filename, 'doc/thesis/tables/', raw=True)

create_table('basement', 'shallow_basement.txt')
create_table('basement', 'deep_basement.txt', 3)
create_table('iros', 'shallow_iros.txt')
create_table('iros', 'deep_iros.txt', 3)
create_table_all('basement', 'all_basement.txt')
create_table_all('iros', 'all_iros.txt', 3)
create_table_diff('diff_iros.txt')

