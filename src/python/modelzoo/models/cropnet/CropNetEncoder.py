from modelzoo.models.Encoder import Encoder
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
import numpy as np

from utils.labels.utils import resize_label


class CropNetEncoder(Encoder):
    def __init__(self, grid_shape, input_shape, encoding):
        self.encoding = encoding
        self.input_shape = input_shape
        self.grid_shape = grid_shape

    def encode_label(self, label: ImgLabel) -> np.array:
        if self.encoding is 'grid':
            return self._encode_grid(label)
        elif self.encoding is 'anchor':
            return self._encode_anchor(label)
        else:
            raise ValueError('Unknown Encoding')

    def _encode_anchor(self, label: ImgLabel) -> np.array:
        #TODO
    def _encode_grid(self, label: ImgLabel) -> np.array:
        label_t = np.zeros(self.input_shape)

        for obj in label.objects:
            label_t[int(np.ceil(self.input_shape[0] - obj.y_max)):int(np.ceil(self.input_shape[0] - obj.y_min)),
            int(np.ceil(obj.x_min)): int(np.ceil(obj.x_max))] = 1.0

        label_t = resize(Image(label_t, 'bgr'), self.grid_shape).array

        return label_t

    def encode_img(self, image: Image) -> np.array:
        return np.expand_dims(image.array, axis=0)
