from sklearn.cluster import KMeans

from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from utils.BoundingBox import BoundingBox
from utils.fileaccess.GateGenerator import GateGenerator
import numpy as np

from utils.fileaccess.utils import save_file
from utils.workdir import work_dir


def generate_anchors(boxes_wh, n_anchors):
    kmeans = KMeans(n_clusters=n_anchors).fit(boxes_wh)
    # print("Centers", kmeans.cluster_centers_)

    centers = kmeans.cluster_centers_
    distances = np.zeros((n_samples, 1))
    for i in range(n_samples):
        j = kmeans.predict(boxes_wh[i].reshape(-1, 2))
        w = np.minimum(centers[j, 0], boxes_wh[i, 0])
        h = np.minimum(centers[j, 1], boxes_wh[i, 1])
        intersect = w * h
        union = centers[j, 0] * centers[j, 1] + boxes_wh[i, 0] * boxes_wh[i, 1] - intersect

        distances[i] = 1.0 - intersect / union

    return kmeans, distances


work_dir()
generator = GateGenerator(directories=['resource/ext/samples/bebop20k/'],
                          batch_size=10, color_format='bgr',
                          shuffle=True, start_idx=0, valid_frac=0,
                          label_format='xml')
boxes_all = []
idx = 0
n_samples = generator.n_samples
range_anchors = 20
for batch in generator.generate():
    for img, label, _ in batch:
        boxes = BoundingBox.from_label(label)
        boxes_all.extend(boxes)
    if idx > n_samples:
        break
    idx += 1

box_dim = BoundingBox.to_tensor_centroid(boxes_all)[:, 2:]
n_anchors = np.arange(1, range_anchors + 1)

clusters = []
mean_dists = np.zeros((range_anchors,))
std_dists = np.zeros((range_anchors,))
for i in range(range_anchors):
    kmeans, distances = generate_anchors(box_dim, i + 1)
    clusters.append(kmeans)
    mean_dists[i] = np.mean(distances)
    std_dists[i] = np.std(distances)

max_iou = np.argmin(mean_dists)
best_tradeoff = np.argmax((1 - mean_dists) / n_anchors)

print("Highest IoU with {} at dim: {}".format(max_iou + 1, clusters[max_iou].cluster_centers_))
print("Best Tradeoff with {} at dim: {}".format(best_tradeoff + 1, clusters[best_tradeoff].cluster_centers_))

plot = BasePlot(x_data=n_anchors, y_data=1 - mean_dists,
                y_label='Average IOU', x_label='Number of Anchors',
                line_style='x--')
plot.save('kmeans_anchors_bebop.png')
save_file(clusters, 'kmeans_clusters_bebop.pkl')
plot.show(True)
