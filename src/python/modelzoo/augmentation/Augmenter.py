from abc import ABC

from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class Augmenter(ABC):
    def augment(self, img: Image, label: ImgLabel):
        pass
