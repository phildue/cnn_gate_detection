import numpy as np
from utils.BoundingBox import BoundingBox

from modelzoo.models.Encoder import Encoder
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from utils.imageprocessing.Backend import normalize
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class YoloEncoder(Encoder):
    def __init__(self, anchor_dims, img_norm, grids, n_boxes, n_classes):

        self.anchor_dims = anchor_dims
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.grids = grids
        self.norm = img_norm

    def _assign_true_boxes(self, anchors, true_boxes):
        coords = anchors.copy()
        confidences = np.zeros((anchors.shape[0], 1)) * np.nan
        class_prob = np.zeros((anchors.shape[0], self.n_classes))

        anchor_boxes = BoundingBox.from_tensor_centroid(confidences, coords)

        for b in true_boxes:
            max_iou = 0.0
            match_idx = np.nan
            b.cy = self.norm[0] - b.cy
            for i, b_anchor in enumerate(anchor_boxes):
                iou = b.iou(b_anchor)
                if iou > max_iou and np.isnan(confidences[i]):
                    max_iou = iou
                    match_idx = i

            if np.isnan(match_idx):
                print("\nGateEncoder::No matching anchor box found!::{}".format(b))
            else:
                confidences[match_idx] = 1.0
                class_prob[match_idx, b.prediction] = 1.0
                coords[match_idx] = b.cx, b.cy, b.w1, b.h1

        confidences[np.isnan(confidences)] = 0.0
        return confidences, class_prob, coords

    def _encode_coords(self, anchors_assigned, anchors):
        wh = anchors_assigned[:, -2:] / anchors[:, -2:]
        c = anchors_assigned[:, -4:-2] - anchors[:, -4:-2]
        c /= anchors[:, -2:]
        return np.hstack((c, wh))

    def encode_img(self, image: Image):
        # TODO do we need this normalization?
        img = normalize(image)
        return np.expand_dims(img.array, axis=0)

    def encode_label(self, label: ImgLabel):
        """
        Encodes bounding box in ground truth tensor.

        :param label: image label containing objects, their bounding boxes and names
        :return: label-tensor

        """
        anchors = GateNetEncoder.generate_encoding(self.norm, self.grids, self.anchor_dims, 4)
        objectness, class_prob, coords = self._assign_true_boxes(anchors, BoundingBox.from_label(label))
        coords = self._encode_coords(coords, anchors)

        label_t = YoloEncoder.concat_t(objectness, class_prob, coords, anchors)
        label_t = np.reshape(label_t, (-1, 1 + self.n_classes + 4 + 4))
        return label_t

    @staticmethod
    def concat_t(objectness, class_prob, coords, anchors):
        return np.hstack((objectness, class_prob, coords, anchors))
