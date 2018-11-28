import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workdir import cd_work

cd_work()
models = [
    'sign',
    'ewfo',
    'cats',
    'sign_deep',
    'ewfo_deep',
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
    print(frame.to_string())
    frames.append(frame)

bins = frames[0]['Sizes Bins']

frame_result = pd.DataFrame()

frame_result['Names'] = models

columns = ['sign', 'ewfo', 'cats']

for c in columns:
    column_content = []
    for i_m, m in enumerate(models):
        result = frames[i_m]['test_basement_' + c]
        column_content.append(result)
    frame_result[c] = column_content
