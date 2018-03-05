import numpy as np
from frontend.utils.BoundingBox import BoundingBox

from src.python.modelzoo.models.Decoder import Decoder


class YoloDecoder(Decoder):
    def __init__(self,
                 norm=(416, 416),
                 grid=(13, 13),
                 class_names=None):
        self.n_classes = len(class_names)
        self.grid = grid
        self.norm = norm

    def decode_netout_to_boxes(self, label_t):
        """
        Convert label tensor to objects of type Box.
        :param label_t: y as fed for learning
        :return: boxes
        """
        label_t = np.reshape(label_t, [self.grid[0], self.grid[1], -1, self.n_classes + 5])
        coord_t = label_t[:, :, :, :4]
        class_t = label_t[:, :, :, 5:]
        conf_t = label_t[:, :, :, 4]
        coord_t_dec = self.decode_coord(coord_t)
        coord_t_dec = np.reshape(coord_t_dec, (-1, 4))
        class_t = np.reshape(class_t, (-1, self.n_classes))
        conf_t = np.reshape(conf_t, (-1, 1))
        boxes = BoundingBox.from_tensor_centroid(class_t, coord_t_dec, conf_t)

        return boxes

    def decode_coord(self, coord_t):
        coord_t_dec = coord_t.copy()
        offset_y, offset_x = np.mgrid[:self.grid[1], :self.grid[0]]

        offset_x = np.expand_dims(offset_x, -1)
        offset_x = np.tile(offset_x, (1, 1, 5))

        offset_y = np.expand_dims(offset_y, -1)
        offset_y = np.tile(offset_y, (1, 1, 5))

        coord_t_dec[:, :, :, 0] += offset_x
        coord_t_dec[:, :, :, 1] += offset_y
        coord_t_dec[:, :, :, 0] *= (self.norm[1] / self.grid[1])
        coord_t_dec[:, :, :, 2] *= (self.norm[1] / self.grid[1])
        coord_t_dec[:, :, :, 1] *= (self.norm[0] / self.grid[0])
        coord_t_dec[:, :, :, 3] *= (self.norm[0] / self.grid[0])

        # TODO get rid of this
        coord_t_dec[:, :, :, 1] = self.norm[1] - coord_t_dec[:, :, :, 1]

        return coord_t_dec

    def decode_netout_to_label(self, label_t):
        """
        Convert y as fed for learning to an object of class ImgLabel. The output can be displayed using annotate_bounding_box
        :param label_t: y-tensor as fed for learning (13,13,125)
        :return: label containing objects bounding box and names
        """
        boxes = self.decode_netout_to_boxes(label_t)
        return BoundingBox.to_label(boxes)
