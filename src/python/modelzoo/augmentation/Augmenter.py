from abc import ABC

from labels.ImgLabel import ImgLabel

from src.python.utils.imageprocessing import Image


class Augmenter(ABC):
    def augment(self, img: Image, label: ImgLabel):
        pass
