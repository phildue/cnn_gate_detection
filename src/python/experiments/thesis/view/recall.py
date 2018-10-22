import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelzoo.evaluation.evalcluster import evalcluster_yaw_dist
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

result_files = [
    'out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    # 'out/thesis/datagen/yolov3_gate_varioussim416x416_i01/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_gate_dronemodel416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_allview416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl'
]

titles = [
    'Random Placement',
    'Race Track',
    'Combined',
]

ObjectLabel.classes = ['gate']
dist_bins = 10
angle_bins = 10
angles = np.linspace(0, 180, angle_bins)
distances = np.linspace(0, 12, dist_bins)
ious = [0.4, 0.6, 0.8]
frame = pd.DataFrame()
frame['Name'] = titles
for iou in ious:
    recalls = []
    for i, f in enumerate(result_files):
        result_file = load_file(f)
        labels_pred = result_file['labels_pred']
        labels_true = result_file['labels_true']
        img_files = result_file['image_files']
        fn, tp = evalcluster_yaw_dist(labels_true, labels_pred, conf_thresh=0.5, n_bins_angle=angle_bins,
                                      n_bins_dist=dist_bins,
                                      iou_thresh=iou)

        recall = (tp / (fn + tp))
        recall[np.isnan(recall)] = 0.0
        recalls.append(recall)
    frame['Recall' + str(iou)] = recalls

print(frame.to_string())
recalls = frame['Recall0.6']
plt.figure(figsize=(9, 3))
plt.title('Recall per Cluster')
for i, r in enumerate(recalls):
    plt.subplot(1, 3, i + 1)
    plt.pcolor(r, cmap=plt.cm.viridis, vmin=0, vmax=1.0)
    if i < 3:
        plt.title(titles[i], fontsize=12)

    plt.xlabel('Relative Yaw Angle')
    plt.ylabel('Relative Distance')
    plt.yticks(np.arange(1, dist_bins, 2), np.round(distances[1::2]))
    plt.xticks(np.arange(2, angle_bins, 3), np.round(angles[2::3]))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)

plt.colorbar()
plt.savefig('doc/thesis/fig/recall_yaw.png')

recalls08 = frame['Recall0.8']
recalls06 = frame['Recall0.6']
recalls04 = frame['Recall0.4']
recalls = [recalls04, recalls06, recalls08]
recall_front = []
recall_yaw = []
for r in recalls:
    r_front_iou = []
    r_yaw_iou = []
    for i in range(len(r)):
        r_front_iou.append( (r[i][:, 0] + r[i][:, -1]) / 2 )
        r_yaw_iou.append(np.mean(r[i][:, 1:-1], 1))
    recall_yaw.append(r_yaw_iou)
    recall_front.append(r_front_iou)

plt.figure(figsize=(8, 3))

for j, r in enumerate(recall_front[1:], 1):
    plt.subplot(1, 2, j)
    plt.title('Recall at Yaw +- 20°C, IoU={}'.format(ious[j]))
    plt.xlim(-0.5, 10)
    plt.ylim(0, 0.5)
    w = 1 / len(result_files)
    # colors = [['#faa083', '#e04a1b', '#b33b16'], ['#ffdc8b', '#ffc12e', '#d18317'], ['#c4de8b', '#8abe17', '#466500']]
    for i in range(len(result_files)):
        plt.bar(distances - w + i * w, r[i], width=w, align='center')
    plt.xlabel('Relative Distance [m]')
    plt.ylabel('Recall')
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
plt.legend(titles)
plt.savefig('doc/thesis/fig/recall_front.png')

plt.figure(figsize=(8,3))
for j, r in enumerate(recall_yaw[1:], 1):
    plt.subplot(1, 2, j)
    plt.title('Recall at Yaw  > 20 °C, IoU={}'.format(ious[j]))
    plt.xlim(-0.5, 10)
    plt.ylim(0, 0.5)
    w = 1 / len(ious)
    # colors = [['#faa083', '#e04a1b', '#b33b16'], ['#ffdc8b', '#ffc12e', '#d18317'], ['#c4de8b', '#8abe17', '#466500']]
    for i in range(len(result_files)):
        plt.bar(distances - w + i * w, r[i], width=w, align='center')
    plt.xlabel('Relative Distance [m]')
    plt.ylabel('Recall')
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.4, hspace=0.4)
plt.legend(titles)
plt.savefig('doc/thesis/fig/recall_angle.png')

plt.show(True)
