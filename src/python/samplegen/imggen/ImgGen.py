from abc import ABC, abstractmethod

from labels.ImgLabel import ImgLabel

from src.python.utils.imageprocessing.Image import Image


class ImgGen(ABC):
    @abstractmethod
    def generate(self, shots: [Image], labels: [ImgLabel], n_backgrounds=10) -> (
            [Image], [ImgLabel]):
        pass
