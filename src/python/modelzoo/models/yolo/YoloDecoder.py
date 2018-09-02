import numpy as np

from modelzoo.models.Decoder import Decoder
from utils.BoundingBox import BoundingBox


class YoloDecoder(Decoder):
    def __init__(self,
                 norm,
                 grid,
                 n_classes):
        self.n_classes = n_classes
        self.grid = grid
        self.norm = norm

    def decode_netout_to_boxes(self, label_t):
        """
        Convert label tensor to objects of type Box.
        :param label_t: y as fed for learning
        :return: boxes
        """
        coord_t = label_t[:, self.n_classes + 1:]
        class_t = label_t[:, :self.n_classes]
        conf_t = label_t[:, 1]
        coord_t_dec = self.decode_coord(coord_t)
        coord_t_dec = np.reshape(coord_t_dec, (-1, 4))
        class_t = np.reshape(class_t, (-1, self.n_classes))
        boxes = BoundingBox.from_tensor_centroid(class_t, coord_t_dec, conf_t)

        return boxes

    def decode_coord(self, coord_t):
        """
        Decode the coordinates of the bounding boxes from the label tensor to absolute coordinates in the image.
        :param coord_t: label tensor (only coordinates)
        :return: label tensor in absolute coordinates
        """
        coord_t_dec = coord_t.copy()

        coord_t_dec[:, 0] *= coord_t_dec[:, -2]
        coord_t_dec[:, 2] *= coord_t_dec[:, -2]
        coord_t_dec[:, 1] *= coord_t_dec[:, -1]
        coord_t_dec[:, 3] *= coord_t_dec[:, -1]

        coord_t_dec[:, 0] += coord_t_dec[:, -4]
        coord_t_dec[:, 1] += coord_t_dec[:, -3]
        coord_t_dec[:, 1] = self.norm[0] - coord_t_dec[:, 1]

        return coord_t_dec[:, :4]

    def decode_netout_to_label(self, label_t):
        """
        Convert label tensor to an object of class ImgLabel
        :param label_t: label-tensor as fed for learning
        :return: label containing objects bounding box and names
        """
        boxes = self.decode_netout_to_boxes(label_t)
        return BoundingBox.to_label(boxes)
