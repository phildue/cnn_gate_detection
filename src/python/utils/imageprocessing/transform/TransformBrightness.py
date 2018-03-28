import copy
import random

from utils.imageprocessing.Backend import brightness
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformBrightness(ImgTransform):
    def __init__(self, scale):
        self.scale = scale

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug = brightness(img_aug, self.scale)

        return img_aug, label_aug
