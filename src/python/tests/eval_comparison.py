import numpy as np
import pandas as pd

from evaluation.evaluation import load_predictions, evalcluster_size_ap
from utils.ModelSummary import ModelSummary
from utils.imageprocessing.Backend import imread
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

import argparse

show_t = -1
parser = argparse.ArgumentParser()
parser.add_argument('--show', metavar='s', type=int, default=show_t)
args = parser.parse_args()
show_t = args.show

cd_work()
models = [
    # 'sign',
    # 'cats',
    # 'ewfo',
    'sign_deep',
    'cats_deep',
    'ewfo_deep',
]

datasets = [
    'test_basement_cats',
    'test_basement_gate',
    'test_basement_sign',
    'test_iros_cats',
    'test_iros_gate',
    'test_iros_sign',
]
titles = models

ObjectLabel.classes = ['gate']
n_iterations = 2
size_bins = np.array([0.0, 1.0])
# size_bins = np.array([0.0, 0.125, 0.25, 1.024])
# size_bins = np.array([0.001, 0.002, 0.004, 0.016, 0.032])
for i_m, m in enumerate(models):
    frame = pd.DataFrame()
    for i_d, d in enumerate(datasets):
        for it in range(n_iterations):
            model_dir = 'out/{0:s}_i{1:02d}'.format(m, it)
            try:
                summary = ModelSummary.from_file(model_dir + '/summary.pkl')

                frame['Name'] = [titles[i_m]] * (len(size_bins) - 1)
                frame['Sizes Bins'] = list(size_bins[:-1])
                for iou in [0.4, 0.6, 0.8]:
                    prediction_dir = model_dir + '/test_{}'.format(d)
                    labels_true, labels_pred, img_files = load_predictions(
                        '{}/predictions.pkl'.format(prediction_dir))

                    if show_t >= 0:
                        images = [imread(f, 'bgr') for f in img_files]
                    else:
                        images = None

                    result_size_ap, true_objects_bin = evalcluster_size_ap(labels_true, labels_pred,
                                                                           bins=size_bins * 416 ** 2,
                                                                           images=images, show_t=show_t, iou_thresh=iou)

                    frame['{} Objects'.format(d)] = true_objects_bin
                    frame['{2:s}_ap{0:02f}_i{1:02d}'.format(iou, it, d)] = result_size_ap
                    print(frame.to_string())
                    frame.to_pickle('{}/results_total.pkl'.format(prediction_dir))
                    frame.to_excel('{}/results_total.xlsx'.format(prediction_dir))
            except FileNotFoundError as e:
                print(e)
                continue
