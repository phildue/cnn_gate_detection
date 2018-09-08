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

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def decode_netout_to_boxes(self, label_t):
        """
        Convert label tensor to objects of type Box.
        :param label_t: y as fed for learning
        :return: boxes
        """
        conf_t, class_t, coord_t, anchors_t = YoloDecoder.split_t(label_t)

        class_t = YoloDecoder.softmax(class_t)
        conf_t = YoloDecoder.sigmoid(conf_t)

        coord_t_dec = self.decode_coord(np.vstack((coord_t, anchors_t)))
        coord_t_dec = np.reshape(coord_t_dec, (-1, 4))
        class_t = np.reshape(class_t, (-1, self.n_classes))
        boxes = BoundingBox.from_tensor_centroid(conf_t, coord_t_dec)

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

    @staticmethod
    def split_t(label_t):
        coord_t = label_t[:, -8:-4]
        conf_t = label_t[:, :1]
        class_t = label_t[:, 1:-8]
        anchors_t = label_t[:, -4:]

        return conf_t, class_t, coord_t, anchors_t
