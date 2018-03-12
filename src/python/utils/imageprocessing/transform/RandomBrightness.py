import copy
import random

from utils.imageprocessing.Backend import brightness
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RandomBrightness(ImgTransform):
    def __init__(self, b_min=0.1, b_max=2.0):
        self.b_max = b_max
        self.b_min = b_min

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        scale = random.uniform(self.b_min, self.b_max)

        img_aug = brightness(img_aug, scale)

        return img_aug, label_aug
