import numpy as np

from modelzoo.models.Decoder import Decoder
from modelzoo.models.cropnet import CropNetAnchorDecoder


class CropNetDecoder(Decoder):
    def decode_netout_to_label(self, netout):
        if self.encoding == 'anchor':
            return self.decoder.decode_netout_to_label(netout)
        else:
            return netout

    def decode_netout_to_boxes(self, netout):
        if self.encoding == 'anchor':
            self.decoder.decode_netout_to_boxes(netout)
        else:
            return netout

    def decode_coord(self, coord_t) -> np.array:
        if self.encoding == 'anchor':
            return self.decoder.decode_coord(coord_t)
        else:
            return coord_t

    def __init__(self, grid_shape, img_shape, encoding):
        self.encoding = encoding
        if encoding == 'anchor':
            self.decoder = CropNetAnchorDecoder(norm=img_shape, grid=grid_shape)
