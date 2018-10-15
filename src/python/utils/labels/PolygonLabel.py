import copy

from utils.Polygon import Polygon
from utils.labels.Pose import Pose


class PolygonLabel:
    classes = []

    def __init__(self, poly: Polygon, class_name: str, pose: Pose = None):
        self.class_name = class_name
        self.poly = poly
        self.pose = pose

    @staticmethod
    def name_to_id(name: str) -> int:
        return PolygonLabel.classes.index(name) + 1

    @staticmethod
    def id_to_name(id: int) -> str:
        try:
            return PolygonLabel.classes[id - 1]
        except IndexError:
            return "Unknown"

    def __repr__(self):
        return '{0:s}: \t{1:s}'.format(self.class_name, self.poly)

    @property
    def mat(self):
        return self.poly.coords_centroid

    @property
    def class_id(self):
        return self.name_to_id(self.class_name)

    def copy(self):
        return copy.deepcopy(self)
