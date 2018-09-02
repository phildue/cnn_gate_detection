import numpy as np

from modelzoo.models.Encoder import Encoder
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Backend import normalize
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel


class YoloEncoder(Encoder):
    def __init__(self, anchor_dims, img_norm, grids, n_boxes, n_classes):
        self.n_classes = n_classes

        self.anchor_dims = anchor_dims
        self.n_boxes = n_boxes
        self.grids = grids
        self.norm = img_norm

    @staticmethod
    def generate_anchors(norm, grids, anchor_dims):

        n_output_layers = len(grids)
        anchors = [YoloEncoder.generate_anchor_layer(norm, grids[i], anchor_dims[i]) for i in
                   range(n_output_layers)]
        anchors = np.concatenate(anchors, 0)

        return anchors

    @staticmethod
    def generate_anchor_layer(norm, grid, anchor_dims):
        n_boxes = len(anchor_dims)
        anchor_t = np.zeros((grid[0], grid[1], n_boxes, 4)) * np.nan

        cell_height = norm[0] / grid[0]
        cell_width = norm[1] / grid[1]
        cx = np.linspace(cell_width / 2, norm[1] - cell_width / 2, grid[1])
        cy = np.linspace(cell_height / 2, norm[0] - cell_height / 2, grid[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)
        anchor_t[:, :, :, 0] = cx_grid
        anchor_t[:, :, :, 1] = cy_grid

        for i in range(n_boxes):
            anchor_t[:, :, i, 2:4] = np.array(norm) / np.array(grid) / anchor_dims[i]

        anchor_t = np.reshape(anchor_t, (grid[0] * grid[1] * n_boxes, -1))

        return anchor_t

    def _assign_true_boxes(self, anchors, true_boxes):
        coords = anchors.copy()
        confidences = np.zeros((anchors.shape[0], 1)) * np.nan
        class_probs = np.zeros((anchors.shape[0], self.n_classes))
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
                class_probs[match_idx, 0] = 1.0
                coords[match_idx] = b.cx, b.cy, b.w, b.h

        confidences[np.isnan(confidences)] = 0.0
        return class_probs, confidences, coords

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
        anchors = YoloEncoder.generate_anchors(self.norm, self.grids, self.anchor_dims)
        class_probs, confidences, coords = self._assign_true_boxes(anchors, BoundingBox.from_label(label))
        coords = self._encode_coords(coords, anchors)
        label_t = np.hstack((class_probs, confidences, coords, anchors))
        label_t = np.reshape(label_t, (-1, self.n_classes + 1 + 4 + 4))
        return label_t
