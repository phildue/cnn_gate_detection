import copy

import numpy as np
from frontend.augmentation.AugmenterColorShift import AugmenterColorShift
from frontend.augmentation.AugmenterFlip import AugmenterFlip
from imageprocessing.Backend import resize, translate
from labels.ImgLabel import ImgLabel

from src.python.modelzoo.augmentation.Augmenter import Augmenter
from src.python.utils.imageprocessing.Image import Image


class YoloAugmenter(Augmenter):
    def __init__(self):
        self.augmenter_flip = AugmenterFlip()
        self.augmenter_color_shift = AugmenterColorShift()

    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)
        h, w, c = img_aug.array.shape

        scale = np.random.uniform() / 10. + 1.
        img_aug, label_aug = resize(img_aug, scale_x=scale, scale_y=scale, label=label_aug)

        max_offx = (scale - 1.) * w
        max_offy = (scale - 1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)
        img_aug, label_aug = translate(img_aug, shift_x=offx, shift_y=offy, label=label_aug)

        if np.random.binomial(1, .5) > 0.5:
            img_aug, label_aug = self.augmenter_flip.augment(img_aug, label_aug)

        img_aug, label_aug = self.augmenter_color_shift.augment(img_aug, label_aug)

        return img_aug, label_aug
