import numpy as np
# noinspection PyDefaultArgument
from sklearn.neural_network._base import softmax

from modelzoo.models.Decoder import Decoder
from utils.BoundingBox import BoundingBox


class SSDDecoder(Decoder):
    def decode_netout_to_label(self, label_tensor):
        boxes = self.decode_netout_to_boxes(label_tensor)
        return BoundingBox.to_label(boxes)

    def __init__(self, img_shape):
        self.img_width = img_shape[1]
        self.img_height = img_shape[0]

    def decode_coord(self, coord_t):
        coord_decoded_t = np.zeros((coord_t.shape[0], 4))

        anchor_cxy = coord_t[:, -4:-2]
        anchor_wh = coord_t[:, -2:]

        variances = coord_t[:, -8:-4]

        coord_decoded_t[:, 0] = coord_t[:, 0] * anchor_wh[:, 0] / variances[:, 0]
        coord_decoded_t[:, 1] = coord_t[:, 1] * anchor_wh[:, 1] / variances[:, 1]

        coord_decoded_t[:, :2] += anchor_cxy

        coord_decoded_t[:, 2] = np.exp(coord_t[:, 2] * variances[:, 2])
        coord_decoded_t[:, 3] = np.exp(coord_t[:, 3] * variances[:, 3])

        coord_decoded_t[:, 2] *= anchor_wh[:, 0]
        coord_decoded_t[:, 3] *= anchor_wh[:, 1]

        coord_decoded_t[:, 0] *= self.img_width
        coord_decoded_t[:, 2] *= self.img_width
        coord_decoded_t[:, 1] *= self.img_height
        coord_decoded_t[:, 3] *= self.img_height

        return coord_decoded_t

    def decode_netout_to_boxes(self, netout_t, min_conf=0.01):
        """
        Decodes the raw network output to bounding boxes
        :param netout_t: tensor(#boxes,#classes+13) raw network output
        :param min_conf: threshold to accept as valid box this is a prefiltering step
         the overall output is filtered in a later step, default value is from the paper
        :return: list of bounding boxes in absolute image coordinates
        """
        class_t = netout_t[:, :-12 - 1]

        if np.any(np.isnan(netout_t[:, -1])):  # if nan we now its network output not truth encoding
            class_t = softmax(class_t)

        class_t = class_t[:, 1:]
        confidence = np.max(class_t, axis=1)
        mask = (confidence > min_conf)

        coord_t = netout_t[:, -13:-1]
        coord_decoded_t = self.decode_coord(coord_t)

        detections_t = netout_t[mask]
        coord_decoded_t = coord_decoded_t[mask]

        class_t = detections_t[:, 1:-12 - 1]

        boxes = BoundingBox.from_tensor_centroid(class_t, coord_decoded_t)

        return boxes
