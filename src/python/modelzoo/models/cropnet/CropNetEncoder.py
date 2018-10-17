import numpy as np

from modelzoo.models.Encoder import Encoder
from modelzoo.models.cropnet import CropNetAnchorEncoder
from modelzoo.models.cropnet import CropNetGridEncoder
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class CropNetEncoder(Encoder):
    def __init__(self, grid_shape, input_shape, encoding, anchor_scale):
        if encoding == 'grid':
            self.encoder = CropNetGridEncoder(grid_shape, input_shape)
        elif encoding == 'anchor':
            self.encoder = CropNetAnchorEncoder(img_norm=input_shape, grids=grid_shape, anchor_scale=anchor_scale)

    def encode_label(self, label: ImgLabel) -> np.array:
        return self.encoder.encode_label(label)

    def encode_img(self, image: Image) -> np.array:
        return self.encoder.encode_img(image)
