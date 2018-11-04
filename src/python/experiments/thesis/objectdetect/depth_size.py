import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelzoo.evaluation.evalcluster import evalcluster_size_ap
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

result_files = [
    'out/thesis/objectdetect/yolov3_d0_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_d1_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_d2_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_d02_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_d01_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
]
titles = [
    'd0',
    'd1',
    'd2',
    'd02',
    'd01',
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
for i, f in enumerate(result_files):
    result_file = load_file(f)
    labels_pred = result_file['labels_pred']
    labels_true = result_file['labels_true']
    img_files = result_file['image_files']

    ap, true = evalcluster_size_ap(labels_true, labels_pred, n_bins=bins,
                                   iou_thresh=iou, min_size=min_size, max_size=max_size)
    aps.append(ap)
    n_true.append(true)
frame['AveragePrecision' + str(iou)] = aps
frame['Objects'] = n_true

print(frame.to_string())

plt.figure(figsize=(8, 3))
plt.title('AveragePrecision')
w = 1.0 / len(titles)
plt.bar(np.arange(bins), np.array(frame['Objects'][0]) / np.max(np.max(frame['Objects'])), width=1.0, color='gray')
for i, r in enumerate(frame['AveragePrecision' + str(iou)]):
    plt.bar(np.arange(bins) - w + i * w, r, width=w)

    plt.xlabel('Size')
    plt.ylabel('Average Precision')
    plt.xticks(np.arange(bins), np.round(sizes, 2))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
legend = ['N_Objects']
legend.extend(titles)
plt.legend(legend)

plt.savefig('doc/thesis/fig/depth_ap_size.png')

plt.show(True)
