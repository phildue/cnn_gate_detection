import copy

from modelzoo.augmentation.Augmenter import Augmenter
from utils.imageprocessing.Backend import convert_color, COLOR_BGR2GRAY
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class AugmenterGray(Augmenter):
    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug = convert_color(img_aug, COLOR_BGR2GRAY)

        return img_aug, label_aug
