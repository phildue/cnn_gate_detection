import numpy as np

from modelzoo.models.Decoder import Decoder


class CropNetDecoder(Decoder):
    def decode_netout_to_label(self, netout):
        return netout

    def decode_netout_to_boxes(self, netout):
        return netout

    def decode_coord(self, coord_t) -> np.array:
        pass

    def __init__(self, grid_shape, img_shape, conf=0.6):
        self.conf = conf
        self.img_shape = img_shape
        self.grid_shape = grid_shape
