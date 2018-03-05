from abc import ABC, abstractmethod

from labels.ImgLabel import ImgLabel

from src.python.utils.imageprocessing.Image import Image


class Metric(ABC):
    @abstractmethod
    def update(self, label_true: ImgLabel, label_pred: ImgLabel):
        pass

    @property
    @abstractmethod
    def result(self):
        pass

    @property
    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def show_result(self, img: Image):
        pass
