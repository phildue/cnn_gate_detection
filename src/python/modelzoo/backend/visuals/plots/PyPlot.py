from abc import ABC, abstractmethod


class PyPlot(ABC):

    @abstractmethod
    def show(self, block=True):
        pass

    @abstractmethod
    def save(self, filename: str = None):
        pass

    @abstractmethod
    def create_fig(self):
        pass
