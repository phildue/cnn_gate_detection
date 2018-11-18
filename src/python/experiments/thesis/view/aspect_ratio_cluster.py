import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.evalcluster import evalcluster_height_width
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

result_files = [
    'out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    # 'out/thesis/datagen/yolov3_gate_varioussim416x416_i01/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_gate_dronemodel416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_allgen416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl'
]
titles = [
    'Random View',
    'Drone Model',
    'Combined'
]
ObjectLabel.classes = ['gate']
bins = 20
angles = np.linspace(0, 416, bins)
distances = np.linspace(0, 416, bins)
ious = [0.4, 0.6, 0.8]
frame = pd.DataFrame()
frame['Name'] = titles
for iou in ious:
    recalls = []
    precisions = []
    for i, f in enumerate(result_files):
        result_file = load_file(f)
        labels_pred = result_file['labels_pred']
        labels_true = result_file['labels_true']
        img_files = result_file['image_files']

        tp, fp, fn = evalcluster_height_width(labels_true, labels_pred, conf_thresh=0.5, n_bins=20,
                                              iou_thresh=iou)

        recall = (tp / (fn + tp))
        recall[np.isnan(recall)] = 0.0
        recalls.append(recall)

        precision = (tp/(tp+fp))
        precision[np.isnan(precision)] = 0.0
        precisions.append(precision)
    frame['Recall' + str(iou)] = recalls
    frame['Precision' + str(iou)] = precisions

titles = [
    'Trained With Random Placement',
    'Trained Following Race Track',
    'Combined',
]
plt.figure(figsize=(8, 3))
plt.title('Recall per Cluster')
for i, r in enumerate(frame['Recall' + str(0.4)]):
    plt.subplot(1, 3, i + 1)
    plt.pcolor(r, cmap=plt.cm.viridis, vmin=0, vmax=1.0)
    if i < 3:
        plt.title(titles[i], fontsize=12)

    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.yticks(np.arange(1, bins, 2), np.round(distances[1::2]))
    plt.xticks(np.arange(2, bins, 3), np.round(angles[2::3]))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)

plt.colorbar()
plt.savefig('doc/thesis/fig/recall_hw.png')

plt.figure(figsize=(8, 3))
plt.title('Precision per Cluster')
for i, p in enumerate(frame['Precision' + str(0.6)]):
    plt.subplot(1, 3, i + 1)
    plt.pcolor(p, cmap=plt.cm.viridis, vmin=0, vmax=1.0)
    if i < 3:
        plt.title(titles[i], fontsize=12)

    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.yticks(np.arange(1, bins, 2), np.round(distances[1::2]))
    plt.xticks(np.arange(2, bins, 3), np.round(angles[2::3]))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)

plt.colorbar()
plt.savefig('doc/thesis/fig/precision_hw.png')

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
