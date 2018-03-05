from abc import abstractmethod

from src.python.modelzoo.backend.tensor import Metric


class Loss(Metric):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass
