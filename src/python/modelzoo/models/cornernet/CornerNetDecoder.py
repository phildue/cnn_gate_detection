import numpy as np
from utils.BoundingBox import BoundingBox
from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel

from modelzoo.models.Decoder import Decoder
from utils.labels.ImgLabel import ImgLabel


class CornerNetDecoder(Decoder):
    def decode_netout_to_label(self, netout) -> ImgLabel:
        netout_dec = self.decode_coord(netout)
        top_left = netout_dec[0]
        top_right = netout_dec[1]
        bottom_left = netout_dec[2]
        bottom_right = netout_dec[3]
        center = (bottom_left + top_right)/2
        corners = GateCorners(center=center,
                              top_left=top_left,
                              top_right=top_right,
                              bottom_left=bottom_left,
                              bottom_right=bottom_right)

        return ImgLabel([GateLabel(gate_corners=corners)])

    def decode_netout_to_boxes(self, netout) -> [BoundingBox]:
        pass

    def decode_coord(self, coord_t) -> np.array:
        coord_dec_t = np.reshape(coord_t, (self.n_polygon, 2))
        coord_dec_t = coord_dec_t * self.img_shape
        return coord_dec_t

    def __init__(self, img_shape, n_polygon=4):
        self.n_polygon = n_polygon
        self.img_shape = img_shape
