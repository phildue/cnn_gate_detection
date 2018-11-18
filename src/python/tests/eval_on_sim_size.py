import numpy as np
import pandas as pd

from evaluation.evalcluster import evalcluster_size_ap
from utils.ModelSummary import ModelSummary
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.transform.TransformCrop import TransformCrop
from utils.imageprocessing.transform.TransformResize import TransformResize
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()
models = [
    'mavnet_lowres320',
    'mavnet_lowres160',
    'yolo_lowres160',
    'yolo_lowres160_i01',
    'yolov3_width0',
]
titles = [
    'Mavnet320x240',
    'Mavnet160x120',
    'Tiny160x120',
    'Tiny160x120_01',
    'Tiny416x416',
]

preprocessing = [
    [TransformCrop(0, 52, 416, 416 - 52), TransformResize((240, 320))],
    [TransformCrop(0, 52, 416, 416 - 52), TransformResize((120, 160))],
    [TransformCrop(0, 52, 416, 416 - 52), TransformResize((120, 160))],
    [TransformCrop(0, 52, 416, 416 - 52), TransformResize((120, 160))],
    None,
]

img_size = [
    320 * 240,
    160 * 120,
    160 * 120,
    160 * 120,
    416 * 416
]
ObjectLabel.classes = ['gate']
bins = 10
frame = pd.DataFrame()
frame['Name'] = titles
frame['img_size'] = img_size

size_bins = np.array([0.001, 0.002, 0.004, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024])[::2]
# size_bins = np.array([0.001, 0.002, 0.004, 0.016, 0.032])
frame['Sizes Bins'] = [size_bins] * len(titles)


for iou in [0.4, 0.6, 0.8]:
    n_true = []
    n_layers = []
    ap_totals = []
    aps = []
    for i, m in enumerate(models):
        result_file = load_file('out/' + m + '/test_iros2018_course_final_simple_17gates/predictions.pkl')
        labels_pred = result_file['labels_pred']
        labels_true = result_file['labels_true']
        img_files = result_file['image_files']
        summary = ModelSummary.from_file('out/' + m + '/summary.pkl')
        labels_true_pp = []
        images_pp = []
        if preprocessing[i]:
            for i_l, l in enumerate(labels_true):
                img = imread(img_files[i_l], 'bgr')
                for p in preprocessing[i]:
                    img, l = p.transform(img, l)
                labels_true_pp.append(l)
                images_pp.append(img)
        else:
            images_pp = [imread(f, 'bgr') for f in img_files]
            labels_true_pp = labels_true
        ap_size, true = evalcluster_size_ap(labels_true_pp, labels_pred,
                                            bins=size_bins * summary.img_size,
                                            images=images_pp, show_t=1)

        aps.append(ap_size)
        n_true.append(true)

        # sum_r, tp, fp, fn, boxes_true = evalset(labels_true, labels_pred, iou_thresh=iou)
        # mean_pr, mean_rec, std_pr, std_rec = average_precision_recall([sum_r])
        # ap_totals.append(np.mean(mean_pr))

        n_layers.append(summary.max_depth)

    frame['AveragePrecision' + str(iou)] = aps
frame['Objects'] = n_true
frame['Layers'] = n_layers
print(frame.to_string())
frame.to_pickle('out/results/size_sim.pkl')
