import matplotlib.pyplot as plt
import numpy as np

from modelzoo.evaluation.evalcluster import evalcluster_yaw_dist
from utils.fileaccess.utils import load_file
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

cd_work()

result_files = [
    'out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
    'out/thesis/datagen/yolov3_gate_varioussim416x416_i01/test_iros2018_course_final_simple_17gates/predictions.pkl',
    # 'out/thesis/datagen/yolov3_gate_dronemodel416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl',
]

ObjectLabel.classes = ['gate']
dist_bins = 10
angle_bins = 10
angles = np.linspace(0, 180, angle_bins)
distances = np.linspace(0, 12, dist_bins)
recalls = []
for i, f in enumerate(result_files):
    result_file = load_file(f)
    labels_pred = result_file['labels_pred']
    labels_true = result_file['labels_true']
    img_files = result_file['image_files']

    fn, tp = evalcluster_yaw_dist(labels_true, labels_pred, conf_thresh=0.5, n_bins_angle=angle_bins, n_bins_dist=dist_bins,
                                  iou_thresh=0.8)

    recall = (tp / (fn + tp))
    recall[np.isnan(recall)] = 0.0
    recalls.append(recall)

titles = [
    'Trained With Random Placement',
    'Trained Following Race Track',
]
plt.figure(figsize=(8, 3))
plt.title('Recall per Cluster')
for i, r in enumerate(recalls):
    plt.subplot(1, 2, i + 1)
    plt.pcolor(r, cmap=plt.cm.viridis, vmin=0, vmax=1.0)
    if i < 2:
        plt.title(titles[i], fontsize=12)

    plt.xlabel('Relative Yaw Angle')
    plt.ylabel('Relative Distance')
    plt.yticks(np.arange(dist_bins), np.round(distances))
    plt.xticks(np.arange(angle_bins), np.round(angles))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.3, hspace=0.3)

plt.colorbar()
plt.savefig('doc/thesis/fig/recall_yaw.png')

plt.figure(figsize=(4, 3))
plt.title('Recall when facing the object frontally')
plt.bar(distances + 1.5, (recalls[0][:, 0]+recalls[0][:, -1]/2), width=0.5, align='center')
plt.bar(distances + 1, (recalls[1][:, 0]+recalls[0][:, -1]/2), width=0.5, align='center')
plt.legend(titles)
plt.xlabel('Relative Distance [m]')
plt.ylabel('Recall')
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/recall_front.png')

plt.show(True)
