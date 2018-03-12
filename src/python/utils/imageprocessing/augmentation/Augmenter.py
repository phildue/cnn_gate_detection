from abc import ABC, abstractmethod

from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class Augmenter(ABC):
    @abstractmethod
    def augment(self, img: Image, label: ImgLabel):
        pass
