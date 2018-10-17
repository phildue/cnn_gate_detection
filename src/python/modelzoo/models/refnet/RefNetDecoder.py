import numpy as np
from utils.BoundingBox import BoundingBox

from modelzoo.models.Decoder import Decoder
from modelzoo.models.gatenet import GateNetDecoder


class RefNetDecoder(Decoder):
    def __init__(self,
                 norm=(416, 416),
                 grid=(13, 13),
                 n_polygon=4, n_roi=5,
                 crop_size=(52, 52)):
        self.n_roi = n_roi
        self.n_polygon = n_polygon
        self.crop_size = crop_size
        self.norm = norm
        self.decoder = GateNetDecoder(norm=crop_size, grid=grid, n_polygon=n_polygon)

    def decode_netout_to_boxes(self, label_t):
        """
        Convert label tensor to objects of type Box.
        :param label_t: y as fed for learning
        :return: boxes
        """
        coord_t = label_t[:, 1:]
        class_t = label_t[:, :1]
        coord_t_dec = self.decode_coord(coord_t)
        coord_t_dec = np.reshape(coord_t_dec, (-1, self.n_polygon))
        class_t = np.reshape(class_t, (-1, 1))
        boxes = BoundingBox.from_tensor_centroid(class_t, coord_t_dec)

        return boxes

    def decode_coord(self, coord_t):
        """
        Decode the coordinates of the bounding boxes from the label tensor to absolute coordinates in the image.
        :param coord_t: label tensor (only coordinates)
        :return: label tensor in absolute coordinates
        """
        roi_t = coord_t[:, -4:]
        coord_t = np.reshape(coord_t, (self.n_roi, -1, self.n_polygon + 4 + 4))
        coord_t_dec = np.vstack([self.decoder.decode_coord(coord_t[i, :, :-4]) for i in range(self.n_roi)])

        coord_t_dec = np.reshape(coord_t_dec, (-1, self.n_polygon))
        coord_t_dec[:, 0] *= roi_t[:, 2] / self.crop_size[1]
        coord_t_dec[:, 1] *= roi_t[:, 3] / self.crop_size[0]
        coord_t_dec[:, 2] *= roi_t[:, 2] / self.crop_size[1]
        coord_t_dec[:, 3] *= roi_t[:, 3] / self.crop_size[0]

        coord_t_dec[:, 0] += roi_t[:, 0]
        coord_t_dec[:, 1] += roi_t[:, 1]
        # coord_t_dec[:, 1] = self.norm[0] - coord_t_dec[:, 1]

        return coord_t_dec

    def decode_netout_to_label(self, label_t):
        """
        Convert label tensor to an object of class ImgLabel
        :param label_t: label-tensor as fed for learning
        :return: label containing objects bounding box and names
        """
        boxes = self.decode_netout_to_boxes(label_t)
        return BoundingBox.to_label(boxes)
