import copy

import numpy as np


class ObjectLabel:
    classes = []

    # TODO add "shift" that moves the points

    @staticmethod
    def name_to_id(name: str) -> int:
        return ObjectLabel.classes.index(name) + 1

    @staticmethod
    def id_to_name(id: int) -> str:
        try:
            return ObjectLabel.classes[id - 1]
        except IndexError:
            return "Unknown"

    def __init__(self, class_name, bounding_box, confidence=1.0):
        self.confidence = confidence
        self.class_name = class_name
        self.__bounding_box = bounding_box
        if class_name not in ObjectLabel.classes:
            ObjectLabel.classes.append(class_name)

    def __repr__(self):
        return '{0:s}: \t{1!s}->{2!s}, {3:}sq'.format(self.class_name, (self.x_min, self.y_min),
                                                      (self.x_max, self.y_max), self.area)

    @property
    def mat(self):
        return self.__bounding_box

    @mat.setter
    def mat(self, mat):
        self.__bounding_box = mat

    @property
    def x_min(self):
        return np.min(self.__bounding_box[:, 0])

    @property
    def y_min(self):
        return np.min(self.__bounding_box[:, 1])

    @property
    def x_max(self):
        return np.max(self.__bounding_box[:, 0])

    @property
    def y_max(self):
        return np.max(self.__bounding_box[:, 1])

    @property
    def class_id(self):
        return self.name_to_id(self.class_name)

    @property
    def width(self):
        return np.abs(self.x_max - self.x_min)

    @property
    def height(self):
        return np.abs(self.y_max - self.y_min)

    @property
    def cx(self):
        return self.x_min + np.abs(self.x_max - self.x_min) / 2

    @property
    def cy(self):
        return self.y_min + np.abs(self.y_max - self.y_min) / 2

    @property
    def area(self):
        return self.width * self.height

    def copy(self):
        return copy.deepcopy(self)
