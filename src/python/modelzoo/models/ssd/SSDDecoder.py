import numpy as np
# noinspection PyDefaultArgument
from frontend.utils.BoundingBox import BoundingBox
from sklearn.neural_network._base import softmax

from src.python.modelzoo.models.Decoder import Decoder


class SSDDecoder(Decoder):
    def decode_netout_to_label(self, label_tensor):
        """ Considering the large number of boxes generated from our method,
         it is essential to perform non - maximum suppression(nms)
         efficiently during inference.By using a confidence threshold of 0.01,
         we can filter out most boxes.We then apply nms with jaccard overlap of 0.45 per class
         and keep the top 200 detections per image [ssd]"""
        boxes = self.decode_netout_to_boxes(label_tensor)
        # boxes = self.non_max_suppression(boxes, self.iou_thresh)
        return BoundingBox.to_label(boxes)

    def __init__(self, anchor_t,
                 img_shape,
                 confidence_thresh=0.3,
                 iou_threshold=0.45,
                 top_k=200,
                 n_classes=20,
                 variances=[1.0, 1.0, 1.0, 1.0]):
        self.anchors_t = anchor_t
        self.n_classes = n_classes
        self.img_width = img_shape[1]
        self.img_height = img_shape[0]
        self.top_k = top_k
        self.iou_thresh = iou_threshold
        self.conf_thresh = confidence_thresh

        self.variances = variances

    def decode_coord(self, coord_t):
        coord_decoded_t = coord_t.copy()
        anchor_wh = self.anchors_t[:, 2:] / np.array([self.img_width, self.img_height])
        anchor_cxy = self.anchors_t[:, :2] / np.array([self.img_width, self.img_height])

        coord_decoded_t[:, 0] = coord_decoded_t[:, 0] * anchor_wh[:, 0] * self.variances[0]
        coord_decoded_t[:, 1] = coord_decoded_t[:, 1] * anchor_wh[:, 1] * self.variances[1]

        coord_decoded_t[:, :2] = coord_decoded_t[:, :2] + anchor_cxy

        coord_decoded_t[:, 2] = np.exp(coord_decoded_t[:, 2] * self.variances[2])
        coord_decoded_t[:, 3] = np.exp(coord_decoded_t[:, 3] * self.variances[3])

        coord_decoded_t[:, 2] = coord_decoded_t[:, 2] * anchor_wh[:, 0]
        coord_decoded_t[:, 3] = coord_decoded_t[:, 3] * anchor_wh[:, 1]

        coord_decoded_t[:, 0] *= self.img_width
        coord_decoded_t[:, 2] *= self.img_width
        coord_decoded_t[:, 1] *= self.img_height
        coord_decoded_t[:, 3] *= self.img_height

        return coord_decoded_t

    def decode_netout_to_boxes(self, netout_t):
        # 0.01 from the paper
        class_t = netout_t[:, :-4]
        class_t = softmax(class_t)
        class_t = class_t[:, 1:]
        confidence = np.max(class_t, axis=1)
        mask = (confidence > 0.01)

        coord_t = netout_t[:, -4:]
        coord_decoded_t = self.decode_coord(coord_t)

        detections_t = netout_t[mask]
        coord_decoded_t = coord_decoded_t[mask]

        class_t = detections_t[:, 1:-4]

        boxes = BoundingBox.from_tensor_centroid(class_t, coord_decoded_t)

        return boxes
