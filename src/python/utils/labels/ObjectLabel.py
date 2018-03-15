import copy
import numpy as np

class ObjectLabel:
    classes = []

    # TODO add "shift" that moves the points

    @staticmethod
    def name_to_id(name: str) -> int:
        return ObjectLabel.classes.index(name)

    @staticmethod
    def id_to_name(id: int) -> str:
        try:
            return ObjectLabel.classes[id]
        except IndexError:
            return "Unknown"

    def __init__(self, class_name, bounding_box, confidence=1.0):
        self.confidence = confidence
        if class_name not in ObjectLabel.classes:
            ObjectLabel.classes.append(class_name)
        self.class_name = class_name
        self.y_min = bounding_box[0][1]
        self.x_min = bounding_box[0][0]
        self.y_max = bounding_box[1][1]
        self.x_max = bounding_box[1][0]

    def __repr__(self):
        return '{0:s}: \t{1!s}->{2!s}'.format(self.class_name, (self.y_min, self.x_min),
                                              (self.y_max, self.x_max))

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
    def area(self):
        return self.width * self.height

    def copy(self):
        return copy.deepcopy(self)
