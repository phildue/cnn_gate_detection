from abc import ABC, abstractmethod

from utils.labels.ImgLabel import ImgLabel


class FileEvaluator(ABC):
    @abstractmethod
    def evaluate(self, result_file: str):
        pass

    @abstractmethod
    def evaluate_sample(self, label_pred: ImgLabel, label_true: ImgLabel):
        pass
