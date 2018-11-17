import numpy as np

from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show, LEGEND_BOX, COLOR_BLUE, save_labeled
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Polygon import Polygon
from utils.workdir import cd_work

cd_work()

# cluster_centers = load_file('kmeans_clusters_bebop.pkl')[3].cluster_centers_
# print(cluster_centers)
# cluster_centers = np.array([[0.07702505, 0.18822592],
#                             [0.35083306, 0.46027088],
#                             [0.47501832, 0.82105769],
#                             [0.19683103, 0.26281582],
#                             [0.10340291, 0.42767551]])
# n_boxes = cluster_centers.shape[0]

img_res = 416, 416
anchors = np.array([
    [[81, 82],
     [135, 169],
     [344, 319]],
    [[10, 14],
     [23, 27],
     [37, 58]],
])
architecture = [
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 1024, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 256, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'predict'},
    {'name': 'route', 'index': [-4]},
    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 128, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'upsample', 'size': 2},
    {'name': 'route', 'index': [-1, 8]},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1),
     'alpha': 0.1},
    {'name': 'predict'}]

predictor = GateNet.create_by_arch(architecture, norm=(416, 416), anchors=anchors, n_polygon=4)
n_boxes = predictor.n_boxes

anchors_t = GateNetEncoder.generate_encoding(predictor.norm, predictor.grid, predictor.anchors, 4)
anchors_boxes = Polygon.from_quad_t_centroid(anchors_t)
black_image = Image(np.ones(predictor.input_shape) * 255, 'bgr')
for j in range(0, 416, int(416 / 13)):
    if j == 0 : continue
    black_image.array[:, j:j + 2] = 0.5
for i in range(0, 416, int(416 / 13)):
    if i == 0 : continue
    black_image.array[i:i + 2, :] = 0.5

for j in range(0, 416, int(416 / 26)):
    if j == 0 : continue
    black_image.array[:, j:j + 1] = 0.5
for i in range(0, 416, int(416 / 26)):
    if i == 0 : continue
    black_image.array[i:i + 1, :] = 0.5

k = 0

# for j, g in enumerate(predictor.grid):
#     step = len(anchors[j])
#     for i in range(0, g[0] * g[1] * step, step):
#         label = ImgLabel([ObjectLabel('anchor', 1.0, a) for a in anchors_boxes[k:k + step]])
#         k += step
#         show(img=black_image, labels=label, legend=LEGEND_BOX)

labels = [ImgLabel([ObjectLabel('anchor', 0.8, a) for a in anchors_boxes[13 * 3 * 6 + 6 * 3:13 * 3 * 6 + 6 * 3 + 3]]),
          ImgLabel([ObjectLabel('anchor', 0.8, a) for a in
                    anchors_boxes[13 * 13 * 3 + 26 * 3 * 13 + 12 * 3:13 * 13 * 3 + 26 * 3 * 13 + 12 * 3 + 3]])]
save_labeled(black_image, 'doc/thesis/fig/yolov3_anchors', labels=labels, legend=LEGEND_BOX,
             colors=[(0, 0, 255), COLOR_BLUE])

show(img=black_image, labels=labels, legend=LEGEND_BOX, colors=[(0, 0, 255), COLOR_BLUE])

anchors = np.array([[
    [330, 340],
    [235, 240],
    [160, 165]],
    [[25, 40],
     [65, 70],
     [100, 110]]]
)
predictor = GateNet.create_by_arch(architecture, norm=(416, 416), anchors=anchors, n_polygon=4)

anchors_t = GateNetEncoder.generate_encoding(predictor.norm, predictor.grid, predictor.anchors, 4)
anchors_boxes = Polygon.from_quad_t_centroid(anchors_t)
black_image = Image(np.ones(predictor.input_shape) * 255, 'bgr')
for j in range(0, 416, int(416 / 13)):
    if j == 0 : continue
    black_image.array[:, j:j + 2] = 0.5
for i in range(0, 416, int(416 / 13)):
    if i == 0 : continue
    black_image.array[i:i + 2, :] = 0.5

for j in range(0, 416, int(416 / 26)):
    if j == 0 : continue
    black_image.array[:, j:j + 1] = 0.5
for i in range(0, 416, int(416 / 26)):
    if i == 0 : continue
    black_image.array[i:i + 1, :] = 0.5

labels = [ImgLabel([ObjectLabel('anchor', 0.8, a) for a in anchors_boxes[13 * 3 * 6 + 6 * 3:13 * 3 * 6 + 6 * 3 + 3]]),
          ImgLabel([ObjectLabel('anchor', 0.8, a) for a in
                    anchors_boxes[13 * 13 * 3 + 26 * 3 * 13 + 12 * 3:13 * 13 * 3 + 26 * 3 * 13 + 12 * 3 + 3]])]
save_labeled(black_image, 'doc/thesis/fig/gate_anchors', labels=labels, legend=LEGEND_BOX,
             colors=[(0, 0, 255), COLOR_BLUE])

show(img=black_image, labels=labels, legend=LEGEND_BOX, colors=[(0, 0, 255), COLOR_BLUE])
