import copy

from imageprocessing.Backend import flip
from labels.ImgLabel import ImgLabel

from src.python.modelzoo.augmentation.Augmenter import Augmenter
from src.python.utils.imageprocessing.Image import Image


class AugmenterFlip(Augmenter):
    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)
        img_aug, label_aug = flip(img_aug, label_aug, 1)

        return img_aug, label_aug
