from abc import ABC, abstractmethod


class Plot(ABC):
    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self, file_path):
        pass
