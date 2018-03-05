import copy

from modelzoo.augmentation.Augmenter import Augmenter
from utils.imageprocessing.Backend import histogram_eq
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class AugmenterHistEq(Augmenter):
    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)
        img_aug = histogram_eq(img_aug)

        return img_aug, label_aug
