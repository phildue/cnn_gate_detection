import numpy as np

from modelzoo.models.Encoder import Encoder
from utils.BoundingBox import BoundingBox
from utils.imageprocessing.Backend import normalize
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class GateNetEncoder(Encoder):
    def __init__(self, anchor_dims=None, img_norm=(416, 416), grids=None, n_boxes=5, n_polygon=4,
                 color_format='yuv'):
        if anchor_dims is None:
            anchor_dims = [np.array([[1.08, 1.19],
                                     [3.42, 4.41],
                                     [6.63, 11.38],
                                     [9.42, 5.11],
                                     [16.62, 10.52]])]
        if grids is None:
            grids = [(13, 13)]

        self.anchor_dims = anchor_dims
        self.n_polygon = n_polygon
        self.color_format = color_format
        self.n_boxes = n_boxes
        self.grids = grids
        self.norm = img_norm

    def _generate_anchors(self):

        n_output_layers = len(self.grids)
        anchors = [self._generate_anchor_layer(self.grids[i], self.anchor_dims[i]) for i in range(n_output_layers)]

        anchors_t = np.concatenate(anchors, 0)
        return anchors_t

    def _generate_anchor_layer(self, grid, anchor_dims):
        n_boxes = len(anchor_dims)
        anchor_t = np.zeros((grid[0], grid[1], n_boxes, 1 + self.n_polygon)) * np.nan

        cell_height = self.norm[0] / grid[0]
        cell_width = self.norm[1] / grid[1]
        cx = np.linspace(cell_width / 2, self.norm[1] - cell_width / 2, grid[1])
        cy = np.linspace(cell_height / 2, self.norm[0] - cell_height / 2, grid[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)
        anchor_t[:, :, :, 0] = cx_grid
        anchor_t[:, :, :, 1] = cy_grid

        for i in range(n_boxes):
            anchor_t[:, :, i, 2:4] = np.array(self.norm) / np.array(grid)

        anchor_t = np.reshape(anchor_t, (grid[0] * grid[1] * n_boxes, -1))

        return anchor_t

    def _assign_true_boxes(self, anchors, true_boxes):
        anchors_assigned = anchors.copy()
        anchor_boxes = BoundingBox.from_tensor_centroid(anchors[:, 4:], anchors[:, :4])

        for b in true_boxes:
            max_iou = 0.0
            match_idx = np.nan
            b.cy = self.norm[0] - b.cy
            for i, b_anchor in enumerate(anchor_boxes):
                iou = b.iou(b_anchor)
                if iou > max_iou:
                    max_iou = iou
                    match_idx = i

            if np.isnan(match_idx):
                print("GateEncoder::No matching anchor box found!::{}".format(b))
            else:
                anchors_assigned[match_idx, 4] = 1.0
                anchors_assigned[match_idx, :4] = b.cx, b.cy, b.w, b.h

        anchors_assigned[np.isnan(anchors_assigned)] = 0.0

        return anchors_assigned

    def _encode_coords(self, anchors_assigned, anchors):
        anchors_encoded = anchors_assigned.copy()
        # TODO clean this up
        offset_y, offset_x = np.mgrid[:self.grids[0][0], :self.grids[0][1]]
        offset_x = np.expand_dims(offset_x, -1)
        offset_y = np.expand_dims(offset_y, -1)
        anchors_encoded[:, :2] /= np.ceil(np.array(self.norm) / np.array(self.grids[0]))
        anchors_encoded[:, 2:4] /= np.ceil(np.array(self.norm) / np.array(self.grids[0]))

        anchors_encoded = np.reshape(anchors_encoded, (13, 13, 5, 5))
        anchors_encoded[:, :, :, 0] -= offset_x
        anchors_encoded[:, :, :, 1] -= offset_y
        anchors_encoded = np.reshape(anchors_encoded, (13 * 13 * 5, 5))

        return anchors_encoded

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
        anchors = self._generate_anchors()
        label_t = self._assign_true_boxes(anchors, BoundingBox.from_label(label))
        label_t = self._encode_coords(label_t, anchors)

        return label_t
