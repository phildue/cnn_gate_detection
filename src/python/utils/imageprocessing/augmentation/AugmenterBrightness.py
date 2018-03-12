import copy
import random

from utils.imageprocessing.Backend import brightness
from utils.imageprocessing.Image import Image
from utils.imageprocessing.augmentation.Augmenter import Augmenter
from utils.labels.ImgLabel import ImgLabel


class AugmenterBrightness(Augmenter):
    def __init__(self, b_min=0.0, b_max=1.0):
        self.b_max = b_max
        self.b_min = b_min

    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        b_min = random.uniform(self.b_min, self.b_max)
        b_max = random.uniform(b_min, self.b_max)

        img_aug = brightness(img_aug, min=b_min, max=b_max)

        return img_aug, label_aug
