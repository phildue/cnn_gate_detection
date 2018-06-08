from modelzoo.models.Encoder import Encoder
import numpy as np

from utils.imageprocessing.Image import Image
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel


class CornerNetEncoder(Encoder):
    def __init__(self, img_shape, n_polygon=4):
        self.n_polygon = n_polygon
        self.img_shape = img_shape

    def encode_label(self, label: ImgLabel) -> np.array:
        label_t = np.zeros((self.n_polygon, 2))
        if len(label.objects)>1:
            raise ValueError('CornerNet should not work on multiple Gates for now')
        for o in label.objects:
            if not isinstance(o, GateLabel):
                raise ValueError('CornerNet makes only sense with corner labels')

            label_t[0] = o.gate_corners.top_left
            label_t[1] = o.gate_corners.top_right
            label_t[2] = o.gate_corners.bottom_left
            label_t[3] = o.gate_corners.bottom_right
            label_t /= self.img_shape

        label_t = np.reshape(label_t, (self.n_polygon * 2))
        return label_t

    def encode_img(self, image: Image) -> np.array:
        return np.expand_dims(image.array, axis=0)
