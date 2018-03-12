import copy

from utils.imageprocessing.Image import Image
from utils.imageprocessing.augmentation.Augmenter import Augmenter
from utils.labels.ImgLabel import ImgLabel


class AugmenterDistort(Augmenter):
    def __init__(self, dist_model: DistortionModel):
        self.dist_model = dist_model

    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)
        img_aug, label_aug = self.dist_model.distort(img_aug, label_aug)

        return img_aug, label_aug
