import matplotlib.pyplot as plt
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
    'yolov3_width0',
]
titles = [
    'Mavnet320x240',
    'Mavnet160x120',
    'Tiny160x120',
    'Tiny416x416',
]

preprocessing = [
    [TransformCrop(0, 52, 416, 416 - 52), TransformResize((240, 320))],
    [TransformCrop(0, 52, 416, 416 - 52), TransformResize((120, 160))],
    [TransformCrop(0, 52, 416, 416 - 52), TransformResize((120, 160))],
    None,
]

img_size = [
    320*240,
    160*120,
    160*120,
    416*416
]
ObjectLabel.classes = ['gate']
bins = 10
iou = 0.6
frame = pd.DataFrame()
frame['Name'] = titles
aps = []

# min_size = 0
# max_size = 0
# areas = []
# for i, f in enumerate(result_files):
#     result_file = load_file(f)
#     labels_pred = result_file['labels_pred']
#     labels_true = result_file['labels_true']
#
#     for l in labels_pred + labels_true:
#         for obj in l.objects:
#             areas.append(obj.poly.area)
# max_size = max(areas)
# min_size = min(areas)
max_size = 1.0
min_size = 0.01

sizes = np.linspace(0, max_size, bins)
n_true = []
n_layers = []
ap_totals = []
for i, m in enumerate(models):
    result_file = load_file('out/' + m + '/test_iros2018_course_final_simple_17gates/predictions.pkl')
    labels_pred = result_file['labels_pred']
    labels_true = result_file['labels_true']
    img_files = result_file['image_files']
    summary = ModelSummary.from_file('out/' + m + '/summary.pkl')
    labels_true_pp = []
    if preprocessing[i]:
        for i_l, l in enumerate(labels_true):
            img = imread(img_files[i_l], 'bgr')
            for p in preprocessing[i]:
                img, l = p.transform(img, l)
            labels_true_pp.append(l)
    else:
        labels_true_pp = labels_true
    ap_size, true = evalcluster_size_ap(labels_true_pp, labels_pred, n_bins=bins,
                                        iou_thresh=iou, min_size=min_size * img_size[i],
                                        max_size=max_size * img_size[i])

    aps.append(ap_size)
    n_true.append(true)

    # sum_r, tp, fp, fn, boxes_true = evalset(labels_true, labels_pred, iou_thresh=iou)
    # mean_pr, mean_rec, std_pr, std_rec = average_precision_recall([sum_r])
    # ap_totals.append(np.mean(mean_pr))

    n_layers.append(summary.max_depth)

frame['AveragePrecision' + str(iou)] = aps
frame['Objects'] = n_true
frame['Layers'] = n_layers
# frame['AP Total' + str(iou)] = ap_totals

print(frame.to_string())

plt.figure(figsize=(8, 3))
plt.title('Performance for Bins of Different Object Areas')
w = 1.0 / len(titles)
# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
legend = []
for i, r in enumerate(frame['AveragePrecision' + str(iou)]):
    plt.bar(np.arange(bins) - len(titles) * w * 0.5 + i * w, r, width=w)
    legend.append(str(frame['Layers'][i]) + ' Layers')
    plt.xlabel('Area Relative to Image Size')
    plt.ylabel('Average Precision')
    plt.xticks(np.arange(bins), np.round(sizes, 2))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
plt.legend(titles)

plt.savefig('doc/thesis/fig/perf_resolution_size.png')

# plt.figure(figsize=(8, 3))
# plt.title('AveragePrecision')
# w = 1.0 / (2 * len(titles))
# n_true = np.array(frame['Objects'][0])  # / np.sum(frame['Objects'][0])
# legend = []
# w = 0.5
# for i, r in enumerate(frame['AveragePrecision' + str(iou)]):
#     legend.append(titles[i])
#     legend.append(titles[i] + ' balanced')
#     plt.bar(frame['Layers'][i]-0.5*w, frame['AP Total' + str(iou)][i], width=w)
#     plt.bar(frame['Layers'][i] + 0.5 * w, np.sum(r * 1 / n_true), width=w)
#
#     plt.xlabel('Layers')
#     plt.ylabel('Average Precision')
#     plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
#                         wspace=0.4, hspace=0.4)
# plt.legend(legend)

# plt.savefig('doc/thesis/fig/depth_ap_size.png')

plt.show(True)
