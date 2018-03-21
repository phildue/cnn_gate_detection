from abc import ABC, abstractmethod


class Net(ABC):
    @abstractmethod
    def compile(self, params=None, metrics=None):
        pass

    @abstractmethod
    def predict(self, sample):
        pass

    @property
    @abstractmethod
    def backend(self):
        pass

    @backend.setter
    @abstractmethod
    def backend(self, model):
        pass

    @property
    @abstractmethod
    def train_params(self):
        pass
