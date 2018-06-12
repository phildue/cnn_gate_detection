from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.yolo.Yolo import Yolo
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show, LEGEND_BOX
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
predictor = GateNet.create('GateNetSingle', grid=(1, 1),norm=(52,52))
n_boxes = 5

# we mimick the last layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


anchor_boxes_t = predictor.encoder.encode_label_batch([ImgLabel([])])
anchor_boxes_t = np.reshape(anchor_boxes_t, (1, predictor.grid[0], predictor.grid[1], n_boxes, -1))
anchor_boxes_t[:, :, :, :, :2] = sigmoid(anchor_boxes_t[:, :, :, :, :2])
anchor_boxes_t[:, :, :, :, 2:4] = np.exp(anchor_boxes_t[:, :, :, :, 2:4]) \
                                  * np.reshape(predictor.anchors, [1, 1, 1, predictor.n_boxes, 2]) \
                                  * np.array([[[[[predictor.grid[0], predictor.grid[1]]] * n_boxes]]])

anchor_boxes = predictor.decoder.decode_netout_to_boxes(anchor_boxes_t)

black_image = Image(np.zeros(predictor.input_shape), 'bgr')

for i in range(int(np.ceil(n_boxes * predictor.grid[0] * predictor.grid[1] / 2)) - n_boxes, len(anchor_boxes)):
    label = BoundingBox.to_label(anchor_boxes[i:i + n_boxes])
    show(img=black_image, labels=label, legend=LEGEND_BOX)
