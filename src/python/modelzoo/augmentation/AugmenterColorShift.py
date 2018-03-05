import copy

import numpy as np
from imageprocessing.Backend import color_shift
from labels.ImgLabel import ImgLabel

from src.python.modelzoo.augmentation.Augmenter import Augmenter
from src.python.utils.imageprocessing.Image import Image


class AugmenterColorShift(Augmenter):
    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug, label_aug = color_shift(img_aug,
                                         np.array([np.random.uniform(),
                                                   np.random.uniform(),
                                                   np.random.uniform()]),
                                         label_aug)

        img_aug.array = img_aug.array[:, :, ::-1]

        return img_aug, label_aug
