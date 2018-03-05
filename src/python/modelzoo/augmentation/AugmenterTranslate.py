import copy

import numpy as np
from imageprocessing.Backend import translate
from labels.ImgLabel import ImgLabel

from src.python.modelzoo.augmentation.Augmenter import Augmenter
from src.python.utils.imageprocessing.Image import Image


class AugmenterTranslate(Augmenter):
    def __init__(self, t_min=-0.3, t_max=0.3):
        self.t_min = t_min
        self.t_max = t_max

    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)
        h, w, c = img_aug.array.shape

        off_x = int(np.random.uniform(self.t_min, self.t_max) * w)
        off_y = int(np.random.uniform(self.t_min, self.t_max) * h)
        img_aug, label_aug = translate(img_aug, shift_x=off_x, shift_y=off_y, label=label_aug)

        return img_aug, label_aug
