import numpy as np


class Polygon:

    @staticmethod
    def to_tensor_minmax(boxes):
        return np.concatenate([np.expand_dims(b.coords_minmax, 0) for b in boxes], 0)

    @staticmethod
    def to_tensor_centroid(boxes):
        return np.concatenate([np.expand_dims(b.coords_centroid, 0) for b in boxes], 0)

    @staticmethod
    def from_tensor_centroid(coord_t, conf_t=None):
        n_boxes = coord_t.shape[0]
        boxes = []
        for i in range(n_boxes):
            b = Polygon()
            b.coords_centroid = coord_t[i]
            boxes.append(b)

        if len(boxes) > 1:
            return boxes
        else:
            return boxes[0]

    def __init__(self):
        self.cx, self.cy, self.w1, self.h1, self.w2, self.h2 = 0., 0., 0., 0., 0., 0.

    def iou(self, box):
        intersection = self.intersect(box)
        union = (self.x_max - self.x_min) * (self.y_max - self.y_min) + (box.x_max - box.x_min) * (
                box.y_max - box.y_min) - intersection
        if union == 0:
            return 1
        else:
            return intersection / union

    def intersect(self, box):
        width = self._overlap([self.x_min, self.x_max], [box.x_min, box.x_max])
        height = self._overlap([self.y_min, self.y_max], [box.y_min, box.y_max])
        return width * height

    def _overlap(self, interval_a, interval_b):
        a, b = interval_a
        c, d = interval_b
        if c < a:
            if d < a:
                return 0
            else:
                return min(b, d) - a
        else:
            if b < c:
                return 0
            else:
                return min(b, d) - c

    @property
    def area(self):
        return self.w1 * self.h1

    @property
    def coords_minmax(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])

    @property
    def coords_centroid(self):
        return np.array([self.cx, self.cy, self.w1, self.h1, self.w2, self.h2])

    @coords_centroid.setter
    def coords_centroid(self, coords):
        self.cx, self.cy, self.w1, self.h1, self.w2, self.h2 = coords

    @property
    def x_min(self):
        return self.cx - self.w1 / 2

    @property
    def x_max(self):
        return self.cx + self.w1 / 2

    @property
    def y_min(self):
        return self.cy - self.h1 / 2

    @property
    def y_max(self):
        return self.cy + self.h1 / 2

    def __repr__(self):
        return '[({0:.2f},{1:.2f}) --> ({2:.2f},{3:.2f})]'.format(self.x_min, self.y_min, self.x_max,
                                                                  self.y_max)
