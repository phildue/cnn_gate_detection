from modelzoo.models.Encoder import Encoder
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
import numpy as np

from utils.labels.utils import resize_label


class CropNetEncoder(Encoder):
    def __init__(self, grid_shape, input_shape):
        self.input_shape = input_shape
        self.grid_shape = grid_shape

    def encode_label(self, label: ImgLabel) -> np.array:
        label_res = resize_label(label, self.input_shape, self.grid_shape)
        label_t = np.zeros(self.grid_shape)

        for obj in label_res.objects:
            label_t[int(np.ceil(obj.y_min)):int(np.ceil(obj.y_max)),
            int(np.ceil(obj.x_min)): int(np.ceil(obj.x_max))] = 1.0

        return label_t

    def encode_img(self, image: Image) -> np.array:
        return np.expand_dims(image.array, axis=0)
