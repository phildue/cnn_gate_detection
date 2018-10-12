import matplotlib.pyplot as plt
import numpy as np

from modelzoo.evaluation.MetricDetection import MetricDetection
from utils.SetAnalysis import SetAnalysis
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work

dist_bins = 10
angle_bins = 10

cd_work()

# evalset(name='test',
#         result_path='resource/ext/samples/cluster_test/',
#         result_file='predictions.pkl',
#         batch_size=16,
#         model_src='out/thesis/datagen/yolov3_gate_realbg416x416_i00/',
#         preprocessing=None,
#         color_format='bgr',
#         image_source=['resource/ext/samples/cluster_test/'])

result_file = load_file(
    'out/thesis/datagen/yolov3_gate_varioussim416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl')
labels_pred = result_file['labels_pred']
labels_true = result_file['labels_true']
img_files = result_file['image_files']
ObjectLabel.classes = ['gate']
poses = []
for l in labels_true:
    for o in l.objects:
        poses.append(o.pose)
yaws = [np.degrees(p.yaw) for p in poses]
dist = [np.linalg.norm(p.transvec/3) for p in poses]
angles = np.linspace(0, 180, angle_bins)
distances = np.linspace(0, 12, dist_bins)
cluster = np.zeros((dist_bins, angle_bins, 2))
conf_thresh = 0.5

plt.figure(figsize=(8, 3))
plt.title("Histogram of Object Occurences in 3D", fontsize=12)
yaws = np.array([np.degrees(p.yaw) for p in poses])
yaws[yaws < 0] *= -1
yaws[yaws > 180] = 360 - yaws[yaws > 180]
yaws = 180 - yaws
dists = [np.linalg.norm(p.transvec / 3) for p in poses]
plt.hist2d(yaws, dists, bins=10,cmap=plt.cm.viridis, vmin=0, vmax=100)
# plt.ylim(0,12)
plt.colorbar()
plt.xlabel("Relative Yaw Angle")
plt.ylabel("Relative Distance")

poses_2 = []
for i in range(len(labels_true)):
    m = MetricDetection(show_=True, min_box_area=0.0 * 416 * 416, max_box_area=2.0 * 416 * 416, min_aspect_ratio=0,
                        max_aspect_ratio=100.0,iou_thresh=0.6)
    label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
    label_true = labels_true[i]
    img = imread(img_files[i], 'bgr')
    # show(img,labels=[label_pred,label_true])
    m.evaluate(label_true, label_pred)
    # m.show_result(img)
    poses_2.extend([b.pose for b in m.boxes_true])

    for b in m.boxes_fn:
        yaw = np.degrees(b.pose.yaw)
        if yaw < 0:
            yaw *= -1
        if yaw > 180:
            yaw = 360 - yaw
        yaw = 180 - yaw
        dist = np.linalg.norm(b.pose.transvec/3)
        i, j = SetAnalysis.assign_angle_dist_to_bin(yaw, dist, angles, distances)
        cluster[i, j, 1] += 1
    for i_b, b in enumerate(m.boxes_true):
        if i_b not in m.boxes_fn:
            yaw = np.degrees(b.pose.yaw)
            if yaw < 0:
                yaw *= -1
            if yaw > 180:
                yaw = 360 - yaw
            yaw = 180 - yaw
            dist = np.linalg.norm(b.pose.transvec / 3)
            i, j = SetAnalysis.assign_angle_dist_to_bin(yaw, dist, angles, distances)
            cluster[i, j, 0] += 1



true_positives = cluster[:, :, 0]# / (cluster[:, :, 0] + cluster[:, :, 1])
true_positives[np.isnan(true_positives)] = 0.0

plt.figure()
plt.pcolor(true_positives,cmap=plt.cm.viridis, vmin=0, vmax=100)
plt.xlabel('Angle')
plt.ylabel('Distance')
plt.yticks(np.arange(dist_bins), np.round(distances))
plt.xticks(np.arange(angle_bins), np.round(angles))
plt.title('True Positives per Cluster')
plt.colorbar()


plt.figure()
false_negatives = cluster[:, :, 1]# / (cluster[:, :, 0] + cluster[:, :, 1])
false_negatives[np.isnan(false_negatives)] = 0.0
plt.pcolor(false_negatives,cmap=plt.cm.viridis, vmin=0, vmax=100)

plt.xlabel('Angle')
plt.ylabel('Distance')
plt.yticks(np.arange(dist_bins), np.round(distances))
plt.xticks(np.arange(angle_bins), np.round(angles))
plt.title('False Negatives per Cluster')
plt.colorbar()

plt.figure()
true = (cluster[:, :, 0] + cluster[:, :, 1])
plt.pcolor(true,cmap=plt.cm.viridis, vmin=0, vmax=100)

plt.xlabel('Angle')
plt.ylabel('Distance')
plt.yticks(np.arange(dist_bins), np.round(distances))
plt.xticks(np.arange(angle_bins), np.round(angles))
plt.title('True per Cluster')
plt.colorbar()


plt.figure()
recall = (cluster[:, :, 0] /(cluster[:, :, 0] + cluster[:, :, 1]))
recall[np.isnan(recall)] = 0.0

plt.pcolor(recall,cmap=plt.cm.viridis, vmin=0, vmax=1.0)

plt.xlabel('Angle')
plt.ylabel('Distance')
plt.yticks(np.arange(dist_bins), np.round(distances))
plt.xticks(np.arange(angle_bins), np.round(angles))
plt.title('Recall per Cluster')
plt.colorbar()
plt.show(True)
