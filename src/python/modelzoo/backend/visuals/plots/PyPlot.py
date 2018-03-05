from abc import abstractmethod

from utils.Plot import Plot


class PyPlot(Plot):
    @abstractmethod
    def save(self, file_path):
        pass

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def create_fig(self):
        pass
