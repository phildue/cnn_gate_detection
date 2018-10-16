import copy

import numpy as np

from utils.Polygon import Polygon
from utils.labels.Pose import Pose


class ObjectLabel:
    classes = []

    def __init__(self, poly: Polygon, class_t, pose: Pose = None):
        self.class_t = class_t
        self.poly = poly
        self.pose = pose

    @staticmethod
    def name_to_id(name: str) -> int:
        try:
            return ObjectLabel.classes.index(name) + 1
        except ValueError:
            ObjectLabel.classes.append(name)
            print("Added class: gate")
        return ObjectLabel.classes.index(name) + 1

    @staticmethod
    def id_to_name(id: int) -> str:
        try:
            return ObjectLabel.classes[id - 1]
        except IndexError:
            return "Unknown"

    def __repr__(self):
        return '{0:s}: \t{1:s}'.format(self.class_name, str(self.poly))

    @property
    def mat(self):
        return self.poly.coords_centroid

    @property
    def class_id(self):
        return np.argmax(self.class_t)

    @property
    def class_name(self):
        return ObjectLabel.id_to_name(self.class_id)

    @staticmethod
    def one_hot_encoding(class_name: str, confidence=1.0):
        idx = ObjectLabel.name_to_id(class_name)
        t = np.zeros((len(ObjectLabel.classes)+1,))
        t[idx] = confidence
        return t

    @property
    def confidence(self):
        return np.max(self.class_t)

    def copy(self):
        return copy.deepcopy(self)
