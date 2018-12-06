import argparse

import numpy as np
import pandas as pd

from evaluation.evaluation import load_predictions, evalcluster_size_ap
from utils.ModelSummary import ModelSummary
from utils.imageprocessing.Backend import imread
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

show_t = -1
parser = argparse.ArgumentParser()
parser.add_argument('--show', metavar='s', type=int, default=show_t)
args = parser.parse_args()
show_t = args.show

cd_work()
models = [
    # 'depth1',
    # 'depth2',
    # 'depth3',
    'racecourt',
    'vgg'
    # 'width2',
    # 'width3',
    # 'width4',

]

datasets = ['iros2018_course_final_simple_17gates',
            'test_iros_gate',
            ]
titles = models

ObjectLabel.classes = ['gate']
n_iterations = 2
size_bins = np.array([0.0, 1.0])
# size_bins = np.array([0.001, 0.002, 0.004, 0.016, 0.032])
for dataset in datasets:
    for i_m, m in enumerate(models):
        for it in range(n_iterations):
            frame = pd.DataFrame()
            model_dir = 'out/{0:s}_i{1:02d}'.format(m, it)
            try:
                summary = ModelSummary.from_file(model_dir + '/summary.pkl')

                frame['Name'] = [titles[i_m]] * (len(size_bins) - 1)
                frame['Sizes Bins'] = list(size_bins[:-1])
                for iou in [0.4, 0.6, 0.8]:
                    prediction_dir = model_dir + '/test_{}'.format(dataset)
                    labels_true, labels_pred, img_files = load_predictions(
                        '{}/predictions.pkl'.format(prediction_dir))

                    if show_t >= 0:
                        images = [imread(f, 'bgr') for f in img_files]
                    else:
                        images = None
                    size_bins_total = size_bins * 416 ** 2
                    result_size_ap, true_objects_bin, recalls, precisions = evalcluster_size_ap(labels_true, labels_pred,
                                                                                                bins=size_bins_total,
                                                                                                min_ar=0,
                                                                                                max_ar=100.0,
                                                                                                min_obj_size=0.001 * 416 ** 2,
                                                                                                max_obj_size=2.0 * 416 ** 2,
                                                                                                images=images,
                                                                                                show_t=show_t,
                                                                                                iou_thresh=iou)

                    frame['{} Objects'.format(dataset)] = true_objects_bin
                    frame['{2:s}_ap{0:02f}_i{1:02d}'.format(iou, it, dataset)] = result_size_ap
                    frame['{2:s}_r50{0:02f}_i{1:02d}'.format(iou, it, dataset)] = [r[4] for r in recalls]
                    frame['{2:s}_p50{0:02f}_i{1:02d}'.format(iou, it, dataset)] = [p[4] for p in precisions]
                    print(frame.to_string())
                    frame.to_pickle('{}/results_total.pkl'.format(prediction_dir))
                    frame.to_excel('{}/results_total.xlsx'.format(prediction_dir))

            except FileNotFoundError as e:
                print(e)
                continue
