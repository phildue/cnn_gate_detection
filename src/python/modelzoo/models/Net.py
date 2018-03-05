from abc import ABC, abstractmethod


class Net(ABC):
    @abstractmethod
    def compile(self, params, metrics):
        pass

    @abstractmethod
    def predict(self, sample):
        pass

    @property
    @abstractmethod
    def backend(self):
        pass

    @property
    @abstractmethod
    def train_params(self):
        pass
