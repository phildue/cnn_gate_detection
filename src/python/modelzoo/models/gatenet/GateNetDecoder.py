import numpy as np

from modelzoo.models.Decoder import Decoder
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Polygon import Polygon


class GateNetDecoder(Decoder):
    def __init__(self,
                 norm,
                 grid,
                 n_polygon=4):
        self.n_polygon = n_polygon
        self.grid = grid
        self.norm = norm

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def decode_netout(self, label_t):
        """
        Convert label tensor to objects of type Box.
        :param label_t: y as fed for learning
        :return: boxes
        """
        coord_t = label_t[:, 1:]
        class_t = GateNetDecoder.sigmoid(label_t[:, :1])
        coord_t_dec = self.decode_coord(coord_t)
        coord_t_dec = np.reshape(coord_t_dec, (-1, self.n_polygon))
        class_t = np.reshape(class_t, (-1, 1))
        boxes = Polygon.from_quad_t_centroid(coord_t_dec)

        labels = []
        for i, b in enumerate(boxes):
            conf = np.max(class_t[i])
            class_id = np.argmax(class_t[i,:])
            label = ObjectLabel(ObjectLabel.id_to_name(class_id), conf, b)
            labels.append(label)

        return ImgLabel(labels)

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

        return coord_t_dec[:, :self.n_polygon]
