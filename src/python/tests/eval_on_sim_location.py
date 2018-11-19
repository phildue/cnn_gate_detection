import numpy as np
import pandas as pd

from evaluation.evaluation import load_predictions, preprocess_truth, evalcluster_location_ap
from utils.ModelSummary import ModelSummary
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.transform.TransformResize import TransformResize
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()
models = [
    # 'mavnet',
    'mavnet_lowres80',
    'mavnet_lowres160',
    'mavnet_lowres320',
    'mavnet_strides',
    'mavnet_strides3_pool2',
    'mavnet_strides4_pool1',
    # 'yolov3_width0',
    # 'yolov3_width1',
    # 'yolov3_width2',
    # 'yolov3_width3',
    'yolo_lowres160'
]
preprocessing = [
    # None,
    [TransformResize((80, 80))],
    [TransformResize((160, 160))],
    [TransformResize((320, 320))],
    [TransformResize((320, 320))],
    [TransformResize((320, 320))],
    [TransformResize((320, 320))],
    [TransformResize((160, 160))],
    # None,
    # None,
    # None,
    # None,
    [TransformResize((160, 160))],
]

sizes = [
    # 416**2,
    80**2,
    160**2,
    320**2,
    320**2,
    320**2,
    320**2,
    160**2,
    # 416**2,
    # 416**2,
    # 416**2,
    # 416**2,
    160**2,
]

titles = models

ObjectLabel.classes = ['gate']
bins = 10
frame = pd.DataFrame()
n_iterations = 2
size_bins = np.array([0, 0.25, 0.75, 1.0])
# size_bins = np.array([0.001, 0.002, 0.004, 0.016, 0.032])

for i_m, m in enumerate(models):
    for it in range(n_iterations):
        model_dir = 'out/{0:s}_i{1:02d}'.format(m, it)
        try:
            summary = ModelSummary.from_file(model_dir + '/summary.pkl')
        except FileNotFoundError:
            print(FileNotFoundError)
            continue
        frame['Name'] = [titles[i_m]]*(len(size_bins)-1)
        frame['Sizes Bins'] = list(size_bins[:-1])
        for iou in [0.4, 0.6, 0.8]:
            prediction_dir = model_dir + '/test_iros2018_course_final_simple_17gates'
            labels_true, labels_pred, img_files = load_predictions(
                '{}/predictions.pkl'.format(prediction_dir))

            if preprocessing[i_m]:
                images, labels_true = preprocess_truth(img_files, labels_true, preprocessing[i_m])
            else:
                images = [imread(f, 'bgr') for f in img_files]

            result_location_ap, true_objects_bin = evalcluster_location_ap(labels_true, labels_pred,
                                                                           bins=size_bins * size,
                                                                           images=images, show_t=1,iou_thresh=iou)

            frame['Objects'] = true_objects_bin
            frame['AveragePrecision{0:02f}_i{1:02d}'.format(iou, it)] = result_location_ap
            print(frame.to_string())
            frame.to_pickle('{}/results_location_cluster.pkl'.format(prediction_dir))
            frame.to_excel('{}/results_location_cluster.xlsx'.format(prediction_dir))
