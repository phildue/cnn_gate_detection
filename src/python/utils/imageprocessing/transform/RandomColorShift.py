import copy

import numpy as np

from utils.imageprocessing.Backend import color_shift
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RandomColorShift(ImgTransform):
    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug, label_aug = color_shift(img_aug,
                                         np.array([np.random.uniform(),
                                                   np.random.uniform(),
                                                   np.random.uniform()]),
                                         label_aug)

        img_aug.array = img_aug.array[:, :, ::-1]

        return img_aug, label_aug
