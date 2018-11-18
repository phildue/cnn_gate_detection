import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.evalcluster import evalcluster_size_ap
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

models = [
    'out/thesis/objectdetect/yolov3_d02_416x416_i00/',
    'out/thesis/objectdetect/yolov3_d01_416x416_i00/',
    'out/thesis/objectdetect/yolov3_d0_416x416_i00/',
    'out/thesis/objectdetect/yolov3_d1_416x416_i00/',
    'out/thesis/objectdetect/yolov3_d2_416x416_i00/',
    'out/thesis/objectdetect/yolov3_d3_416x416_i00/',
    # 'out/thesis/objectdetect/yolov3_w0_416x416_i00/',
    # 'out/thesis/objectdetect/yolov3_w1_416x416_i00/',
    # 'out/thesis/objectdetect/yolov3_w2_416x416_i00/',
    # 'out/thesis/objectdetect/yolov3_w3_416x416_i00/',
    'out/thesis/datagen/yolov3_blur416x416_i00/',
    # 'out/thesis/objectdetect/yolov3_w01_416x416_i00/',
    'out/thesis/datagen/yolov3_arch2416x416_i00/',
    # 'out/thesis/objectdetect/yolov3_grid416x416_i00/',
    'out/thesis/objectdetect/yolov3_pool416x416_i00/',
    'out/thesis/objectdetect/yolov3_avg_pool416x416_i00/',

]
titles = [
    # 'd02',
    # 'd01',
    # 'd0',
    'd1',
    'd2',
    'd3',
    # 'w0',
    # 'w1',
    # 'w2',
    # 'w3',
    'w4',
    # 'w01',
    'arch',
    # 'grid',
    'pool',
    'avg_pool'
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
max_size = 1.05
min_size = 0.01

sizes = np.linspace(0, max_size, bins)
n_true = []
n_layers = []
ap_totals = []
for i, m in enumerate(models):
    result_file = load_file(m + 'test_iros2018_course_final_simple_17gates/predictions.pkl')
    labels_pred = result_file['labels_pred']
    labels_true = result_file['labels_true']
    img_files = result_file['image_files']

    ap_size, true = evalcluster_size_ap(labels_true, labels_pred, n_bins=bins,
                                        iou_thresh=iou, min_size=min_size, max_size=max_size)

    aps.append(ap_size)
    n_true.append(true)

    # sum_r, tp, fp, fn, boxes_true = evalset(labels_true, labels_pred, iou_thresh=iou)
    # mean_pr, mean_rec, std_pr, std_rec = average_precision_recall([sum_r])
    # ap_totals.append(np.mean(mean_pr))

    summary = load_file(m + 'summary.pkl')
    d = 0
    for layer in summary['architecture']:
        if 'conv' in layer['name']:
            d += 1
    n_layers.append(d)

frame['AveragePrecision' + str(iou)] = aps
frame['Objects'] = n_true
frame['Layers'] = n_layers
# frame['AP Total' + str(iou)] = ap_totals

print(frame.to_string())

plt.figure(figsize=(8, 3))
plt.title('AveragePrecision across Size Bins')
w = 1.0 / len(titles)
# plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.sum(frame['Objects'][0]), width=1.0, color='gray')
legend = []
for i, r in enumerate(frame['AveragePrecision' + str(iou)]):
    plt.bar(np.arange(bins) - len(titles)*w + i * w, r, width=w)
    legend.append(str(frame['Layers'][i])+' Layers')
    plt.xlabel('Size')
    plt.ylabel('Average Precision')
    plt.xticks(np.arange(bins), np.round(sizes, 2))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
plt.legend(legend)



plt.show(True)
