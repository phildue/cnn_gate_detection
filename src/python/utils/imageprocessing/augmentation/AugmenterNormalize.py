from utils.imageprocessing.Backend import normalize
from utils.imageprocessing.Image import Image
from utils.imageprocessing.augmentation.Augmenter import Augmenter
from utils.labels.ImgLabel import ImgLabel


class AugmenterNormalize(Augmenter):
    def augment(self, img: Image, label: ImgLabel):
        return normalize(img), label.copy()
