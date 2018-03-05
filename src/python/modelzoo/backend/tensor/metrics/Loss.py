from abc import abstractmethod

from modelzoo.backend.tensor.metrics.Metric import Metric


class Loss(Metric):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass
