import copy
import random

from imageprocessing.fisheye import fisheye
from labels.ImgLabel import ImgLabel

from src.python.modelzoo.augmentation.Augmenter import Augmenter
from src.python.utils.imageprocessing.Image import Image


class AugmenterDistort(Augmenter):
    def __init__(self, k_mult=0.000001, rand_range=(1, 20)):
        self.k_mult = k_mult
        self.rand_range = rand_range

    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)
        k = random.randint(self.rand_range[0], self.rand_range[1])
        img_aug, label_aug = fisheye(img_aug, k=k * self.k_mult, label=label_aug)

        return img_aug, label_aug
