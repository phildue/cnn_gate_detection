import copy

from utils.imageprocessing.DistortionModel import DistortionModel
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformDistort(ImgTransform):
    def __init__(self, dist_model: DistortionModel):
        self.dist_model = dist_model

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)
        img_aug, label_aug = self.dist_model.distort(img_aug, label_aug)

        return img_aug, label_aug
