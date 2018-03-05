from abc import ABC

from src.python.utils.labels.Pose import Pose


class View(ABC):
    def draw(self, offset: Pose = Pose(0, 0, 0, 0, 0, 0)):
        pass
