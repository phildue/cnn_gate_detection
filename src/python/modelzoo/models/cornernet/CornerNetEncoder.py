import numpy as np

from modelzoo.models.Encoder import Encoder
from utils.imageprocessing.Image import Image
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel


class CornerNetEncoder(Encoder):
    def __init__(self, img_shape, n_polygon=4):
        self.n_polygon = n_polygon
        self.img_shape = img_shape

    def encode_label(self, label: ImgLabel) -> np.array:
        label_t = np.zeros(self.img_shape)
        for o in label.objects:
            if not isinstance(o, GateLabel):
                raise ValueError('CornerNet makes only sense with corner labels')

            for i in range(o.gate_corners.mat.shape[0] - 1):
                if 0 < o.gate_corners.mat[i, 1] < self.img_shape[0] and 0 < o.gate_corners.mat[i, 0] < self.img_shape[
                    1]:
                    label_t[o.gate_corners.mat[i, 1].astype(np.int), o.gate_corners.mat[i, 0].astype(np.int)] = 1.0
        # imshow(Image(label_t * 255, 'bgr'))
        return np.expand_dims(label_t, -1)

    def encode_img(self, image: Image) -> np.array:
        return np.expand_dims(image.array, axis=0)
