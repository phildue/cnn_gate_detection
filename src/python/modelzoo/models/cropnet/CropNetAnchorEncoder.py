import numpy as np

from modelzoo.models.Encoder import Encoder
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class CropNetAnchorEncoder(Encoder):

    def __init__(self, anchor_scale=None, img_norm=(416, 416), grids=None,
                 color_format='yuv'):
        if anchor_scale is None:
            anchor_scale = np.array([[0.5,
                                      1,
                                      2]])
        if grids is None:
            grids = [(13, 13)]

        self.anchor_scale = anchor_scale
        self.color_format = color_format
        self.n_boxes = anchor_scale[0].shape[0]
        self.grids = grids
        self.norm = img_norm

    @staticmethod
    def generate_anchors(norm, grids, anchor_scale):

        n_output_layers = int(np.ceil(len(grids) / 2))

        anchors = [CropNetAnchorEncoder.generate_anchor_layer(norm, grids[i], anchor_scale[i]) for i in
                   range(n_output_layers)]

        anchors_t = np.concatenate(anchors, 0)
        return anchors_t

    @staticmethod
    def generate_anchor_layer(norm, grid, anchor_scale):
        n_boxes = len(anchor_scale)
        anchor_t = np.zeros((grid[0], grid[1], n_boxes, 3)) * np.nan

        cell_size = norm[0] / grid[0]
        cx = np.linspace(cell_size / 2, norm[1] - cell_size / 2, grid[1])
        cy = np.linspace(cell_size / 2, norm[0] - cell_size / 2, grid[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)
        anchor_t[:, :, :, 0] = cx_grid
        anchor_t[:, :, :, 1] = cy_grid

        for i in range(n_boxes):
            anchor_t[:, :, i, 2:3] = norm[0] / grid[0] / anchor_scale[i]

        anchor_t = np.reshape(anchor_t, (grid[0] * grid[1] * n_boxes, -1))

        return anchor_t

    def _assign_true_boxes(self, anchors, true_boxes):
        coords = anchors.copy()
        confidences = np.zeros((anchors.shape[0], 1)) * np.nan
        anchor_boxes = BoundingBox.from_tensor_centroid(confidences, np.hstack((coords, coords[:, -1:])))

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
                print("GateEncoder::No matching anchor box found!::{}".format(b))
            else:
                confidences[match_idx] = 1.0

                size = 1.25 * max((b.w, b.h))

                coords[match_idx] = b.cx, b.cy, size

        confidences[np.isnan(confidences)] = 0.0
        return confidences, coords

    def _encode_coords(self, anchors_assigned, anchors):
        wh = anchors_assigned[:, -1:] / anchors[:, -1:]
        c = anchors_assigned[:, -3:-1] - anchors[:, -3:-1]
        c /= anchors[:, -1:]
        return np.hstack((c, wh))

    def encode_img(self, image: Image):

        return np.expand_dims(image.array, axis=0)

    def encode_label(self, label: ImgLabel):
        """
        Encodes bounding box in ground truth tensor.

        :param label: image label containing objects, their bounding boxes and names
        :return: label-tensor

        """
        anchors = CropNetAnchorEncoder.generate_anchors(self.norm, self.grids, self.anchor_scale)
        confidences, coords = self._assign_true_boxes(anchors, BoundingBox.from_label(label))
        coords = self._encode_coords(coords, anchors)
        label_t = np.hstack((confidences, coords, anchors))
        label_t = np.reshape(label_t, (-1, 1 + 3 + 3))
        return label_t
