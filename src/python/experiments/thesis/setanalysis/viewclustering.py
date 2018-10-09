import matplotlib.pyplot as plt
import numpy as np

from modelzoo.evaluation.MetricDetection import MetricDetection
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import cd_work


def assign(dist, angle):
    dist = np.linalg.norm(dist/3)
    if angle < 0:
        angle = 0 - angle
    if angle > 180:
        angle = 360 - angle
    print(dist)
    print(angle)

    for i_dist in range(dist_step - 1):
        if distances[i_dist] <= dist < distances[i_dist + 1]:
            break
    for i_angle in range(angle_step - 1):

        if angles[i_angle] <= angle < angles[i_angle + 1]:
            break

    return i_dist, i_angle


dist_step = 5
angle_step = 6
angles = np.linspace(0, 180, angle_step)
distances = np.linspace(0, 10, dist_step)
cluster = np.zeros((dist_step, angle_step, 2))
conf_thresh = 0.5
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
    'out/thesis/objectdetect/mavnet_416x416_i00/test_iros2018_course_final_simple_17gates/predictions.pkl')
labels_pred = result_file['labels_pred']
labels_true = result_file['labels_true']
img_files = result_file['image_files']
ObjectLabel.classes = ['gate']
for i in range(len(labels_pred)):
    m = MetricDetection(show_=True, min_box_area=0.01 * 416 * 416, max_box_area=2.0 * 416 * 416, min_aspect_ratio=0.3,
                        max_aspect_ratio=3.0)
    label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
    label_true = labels_true[i]
    img = imread(img_files[i], 'bgr')
    # show(img,labels=[label_pred,label_true])
    m.evaluate(label_true, label_pred)
    # m.show_result(img)

    for b in m.boxes_fn:
        i, j = assign(b.pose.transvec, np.degrees(b.pose.yaw))
        cluster[i, j, 1] += 1
    for i_b, b in enumerate(m.boxes_true):
        if i_b not in m.boxes_fn:
            i, j = assign(b.pose.transvec, np.degrees(b.pose.yaw))
            cluster[i, j, 0] += 1

print(cluster)
recall = cluster[:, :, 0] / (cluster[:, :, 0] + cluster[:, :, 1])
plt.imshow(recall)
plt.xlabel('Angle')
plt.ylabel('Distance')
plt.xticks(np.arange(angle_step), angles)
plt.yticks(np.arange(dist_step), distances)
plt.title('Recall per Cluster')
plt.colorbar()
plt.show(True)
