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
    # 'out/thesis/datagen/yolov3_gate_varioussim416x416_i01/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_d1_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_d2_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_d02_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_d01_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/objectdetect/yolov3_w01_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl'
]
titles = [
    'd0',
    'd1',
    'd2',
    'd02',
    'd01',
    'w01'
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
max_size = 2
min_size = 0.01

sizes = np.linspace(0, max_size, bins)
objects = []
for i, f in enumerate(result_files):
    result_file = load_file(f)
    labels_pred = result_file['labels_pred']
    labels_true = result_file['labels_true']
    img_files = result_file['image_files']

    ap = evalcluster_size_ap(labels_true, labels_pred, n_bins=bins,
                             iou_thresh=iou, min_size=min_size, max_size=max_size)
    aps.append(ap)
frame['AveragePrecision' + str(iou)] = aps

print(frame.to_string())

plt.figure(figsize=(8, 3))
plt.title('AveragePrecision')
w = 1 / len(titles)
for i, r in enumerate(frame['AveragePrecision' + str(iou)]):
    plt.bar(np.arange(bins) - w + i * w, r, width=w)

    plt.xlabel('Size')
    plt.ylabel('Average Precision')
    plt.xticks(np.arange(bins), np.round(sizes,2))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
plt.legend(titles)

# plt.savefig('doc/thesis/fig/precision_hw.png')

# plt.figure(figsize=(5, 3))
# plt.title('Recall at Yaw +- 20°c')
# plt.xlim(-0.5, 10)
# plt.bar(distances - .33, (recalls[0][:, 0] + recalls[0][:, -1] / 2), width=0.33, align='center')
# plt.bar(distances, (recalls[1][:, 0] + recalls[1][:, -1] / 2), width=0.33, align='center')
# plt.bar(distances + .33, (recalls[2][:, 0] + recalls[2][:, -1] / 2), width=0.33, align='center')
# plt.legend(titles, bbox_to_anchor=(0.4, 0.6))
# plt.xlabel('Relative Distance [m]')
# plt.ylabel('Recall')
# plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
#                     wspace=0.4, hspace=0.4)
# plt.savefig('doc/thesis/fig/recall_front.png')

plt.show(True)
