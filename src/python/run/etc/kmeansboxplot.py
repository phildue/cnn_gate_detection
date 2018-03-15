from matplotlib.pyplot import boxplot, show
from sklearn.cluster import KMeans

from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.backend.visuals.plots.BasePlot import BasePlot
from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from utils.BoundingBox import BoundingBox
from utils.fileaccess.GateGenerator import GateGenerator
import numpy as np
from utils.imageprocessing.Backend import resize
from utils.fileaccess.utils import save_file
from utils.workdir import work_dir


def generate_anchors(boxes_wh, n_anchors):
    kmeans = KMeans(n_clusters=n_anchors).fit(boxes_wh)
    # print("Centers", kmeans.cluster_centers_)
    n_boxes = len(boxes_wh)
    centers = kmeans.cluster_centers_
    distances = np.zeros((n_boxes, 1))
    for i in range(n_boxes):
        j = kmeans.predict(boxes_wh[i].reshape(-1, 2))
        bw = np.minimum(centers[j, 0], boxes_wh[i, 0])
        bh = np.minimum(centers[j, 1], boxes_wh[i, 1])
        intersect = bw * bh
        union = centers[j, 0] * centers[j, 1] + boxes_wh[i, 0] * boxes_wh[i, 1] - intersect

        distances[i] = 1.0 - intersect / union

    return kmeans, distances


def plot_iou_vs_center(box_dims, range_anchors):
    clusters = []
    mean_dists = np.zeros((range_anchors,))
    std_dists = np.zeros((range_anchors,))
    n_anchors = np.linspace(0, range_anchors, range_anchors)
    for k in range(range_anchors):
        kmeans, distances = generate_anchors(box_dims, k + 1)
        clusters.append(kmeans)
        mean_dists[k] = np.mean(distances)
        std_dists[k] = np.std(distances)

    iou_c_plot = BasePlot(x_data=n_anchors, y_data=1 - mean_dists,
                          y_label='Average IOU', x_label='Number of Anchors',
                          line_style='x--')
    iou_c_plot.save('kmeans_anchors_bebop.png')
    save_file(clusters, 'kmeans_clusters_bebop.pkl')

    return iou_c_plot


work_dir()
path = 'resource/ext/samples/bebop20k/'
set_analyzer = SetAnalyzer((180, 315), path)

plot_iou_vs_center(set_analyzer.get_box_dims(), 15).show()
