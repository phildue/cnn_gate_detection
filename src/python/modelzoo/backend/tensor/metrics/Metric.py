from abc import ABC, abstractmethod


class Metric(ABC):
    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def compute(self, y_true, y_pred):
        pass
