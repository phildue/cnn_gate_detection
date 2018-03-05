import copy

import numpy as np
from imageprocessing.Backend import resize
from labels.ImgLabel import ImgLabel

from src.python.modelzoo.augmentation.Augmenter import Augmenter
from src.python.utils.imageprocessing.Image import Image


class AugmenterScale(Augmenter):
    def __init__(self, min_scale=1.0, max_scale=2.0):
        self.max = max_scale
        self.min = min_scale

    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        scale = np.random.uniform(self.min, self.max)
        img_aug, label_aug = resize(img_aug, scale_x=scale, scale_y=scale, label=label_aug)

        return img_aug, label_aug
