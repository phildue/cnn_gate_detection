import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.fileaccess.utils import save_file
from utils.workdir import cd_work

cd_work()
models = [
    'ewfo',
    'ewfo_sim',
    'ewfo_voc',
]

datasets = [
    'test_iros_gate',
    'test_iros_cats',
    'test_iros_sign',
]

legend = [
    'Single Background',
    'Simulated Background',
    'VOC Background',
]

title = [
    'Gate',
    'Sign',
    'Cats'
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
    columns = ['gate', 'sign', 'cats']
    table = pd.DataFrame()
    table['Trained/Tested'] = legend

    for i_c, c in enumerate(columns):
        column_content = []
        for i_m, _ in enumerate(columns):
            results = []
            for i_i in range(n_iterations):
                result = frame['test_{}_{}_i0{}'.format(setname, c, i_i)][i_m + shift]
                if result >= 0:
                    results.append(result)
            column_content.append('{:.2f} +- {:.2f}'.format(np.mean(results), np.std(results)))
        table[title[i_c]] = column_content

    print(table.to_string(index=False))
    print(table.to_latex(index=False))
    save_file(table.to_latex(index=False), filename, 'doc/thesis/tables/', raw=True)



create_table('iros', 'sim_vs_voc.txt')

