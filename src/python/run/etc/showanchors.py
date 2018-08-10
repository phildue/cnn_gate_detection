from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from modelzoo.models.yolo.Yolo import Yolo
from utils.BoundingBox import BoundingBox
from utils.Polygon import Quadrangle
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show, LEGEND_BOX, LEGEND_CORNERS
from utils.labels.ImgLabel import ImgLabel
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work
import numpy as np

cd_work()

# cluster_centers = load_file('kmeans_clusters_bebop.pkl')[3].cluster_centers_
# print(cluster_centers)
# cluster_centers = np.array([[0.07702505, 0.18822592],
#                             [0.35083306, 0.46027088],
#                             [0.47501832, 0.82105769],
#                             [0.19683103, 0.26281582],
#                             [0.10340291, 0.42767551]])
# n_boxes = cluster_centers.shape[0]
grid = [(13, 13),
        (7, 7),
        (3, 3),
        (1, 1)]
img_res = 208, 208
anchors = [
    [[1, 1, 1, 1], [1.5, 0.5, 1.5, 0.3]],
    [[1, 1, 1, 1], [1.5, 0.5, 1.5, 0.3]],
    [[1, 1, 1, 1], [1.5, 0.5, 1.5, 0.5]],
    # [[1, 1], [1.5, 0.5]],
    [[1, 1, 1, 1], [1.5, 0.5, 1.5, 0.5], [2.5, 0.25, 2.5, 0.25]]
]

architecture = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'predict'},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'predict'},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'predict'},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},
    {'name': 'predict'},

]

predictor = GateNet.create_by_arch(architecture, norm=(208, 208), anchors=anchors, n_polygon=6)
n_boxes = predictor.n_boxes


# we mimick the last layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


anchors_t = GateNetEncoder.generate_anchors(predictor.norm, predictor.grid, predictor.anchors, 6)
anchors_boxes = Quadrangle.from_tensor_centroid(np.array([[1.0]] * anchors_t.shape[0]), anchors_t)
black_image = Image(np.zeros(predictor.input_shape), 'bgr')
k = 0
for j, g in enumerate(predictor.grid):
    step = 1  # len(anchors[j])
    for i in range(0, g[0] * g[1] * step, step):
        label = Quadrangle.to_label(anchors_boxes[k:k + step])
        k += step
        show(img=black_image, labels=label, legend=LEGEND_CORNERS)
