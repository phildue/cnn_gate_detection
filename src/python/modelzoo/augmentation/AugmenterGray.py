import copy

from imageprocessing.Backend import convert_color, COLOR_BGR2GRAY
from labels.ImgLabel import ImgLabel

from src.python.modelzoo.augmentation.Augmenter import Augmenter
from src.python.utils.imageprocessing.Image import Image


class AugmenterGray(Augmenter):
    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug = convert_color(img_aug, COLOR_BGR2GRAY)

        return img_aug, label_aug
