import numpy as np

from utils.imageprocessing.Imageprocessing import get_bounding_box
from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel


class Quadrangle:
    @staticmethod
    def from_label(label: ImgLabel):
        boxes = []
        for l in label.objects:
            if l is None: continue
            if isinstance(l, GateLabel):
                wTop = abs(l.gate_corners.top_left[0] - l.gate_corners.top_right[0])
                wBottom = abs(l.gate_corners.bottom_left[0] - l.gate_corners.bottom_right[0])
                hLeft = abs(l.gate_corners.top_left[1] - l.gate_corners.bottom_left[1])
                hRight = abs(l.gate_corners.top_right[1] - l.gate_corners.bottom_right[1])
                cx, cy = l.gate_corners.center

            elif isinstance(l, ObjectLabel):
                wTop = l.width
                wBottom = l.width
                hLeft = l.height
                hRight = l.height
                cx = l.x_min + (l.x_max - l.x_min) / 2
                cy = l.y_min + (l.y_max - l.y_min) / 2
            else:
                raise NotImplementedError("Don't know how to handle this label")

            b = Quadrangle()
            b.coords_centroid = cx, cy, wTop, hLeft, wBottom, hRight
            b.class_conf = l.confidence
            boxes.append(b)
        return boxes

    @staticmethod
    def to_label(quadrangles) -> ImgLabel:
        """
        Convert quadrangles to gate labels
        :param quadrangles: objects of type quadrangle
        :return: gate labels
        """
        gate_labels = []
        for quad in quadrangles:
            topleft = int((quad.cx - quad.wTop / 2)), int((quad.cy + quad.hLeft / 2))
            topright = int((quad.cx + quad.wTop / 2)), int((quad.cy + quad.hRight / 2))
            bottomleft = int((quad.cx - quad.wBottom / 2)), int((quad.cy - quad.hLeft / 2))
            bottomight = int((quad.cx + quad.wBottom / 2)), int((quad.cy - quad.hRight / 2))
            center = quad.cx, quad.cy

            gate_corners = GateCorners(top_left=topleft,
                                       top_right=topright,
                                       bottom_left=bottomleft,
                                       bottom_right=bottomight,
                                       center=center)

            gate_labels.append(GateLabel(gate_corners=gate_corners, confidence=quad.class_conf))
        return ImgLabel(gate_labels)

    @staticmethod
    def to_tensor_centroid(boxes):
        return np.concatenate([np.expand_dims(b.coords_centroid, 0) for b in boxes], 0)

    @staticmethod
    def to_class_tensor(boxes):
        return np.concatenate([np.expand_dims(b.class_conf, 0) for b in boxes], 0)

    @staticmethod
    def from_tensor_centroid(class_t, coord_t, conf_t=None):
        n_boxes = class_t.shape[0]
        boxes = []
        for i in range(n_boxes):
            b = Quadrangle()
            b.coords_centroid = coord_t[i]
            b.class_conf = conf_t[i] if conf_t is not None else np.max(class_t[i])
            boxes.append(b)
        return boxes

    def __init__(self):
        self._x, self._y, self.wTop, self.hLeft, self.wBottom, self.hRight, self._c = 0., 0., 0., 0., 0., 0., 0.
        self.class_conf = 0

    def iou(self, diamond):
        intersection = self.intersect(diamond)
        union = self.wMax * self.hMax + diamond.wMax * diamond.hMax - intersection
        if union == 0:
            return 1
        else:
            return intersection / union

    @property
    def prediction(self):
        return 1 if self.class_conf >= 0.5 else 0

    def intersect(self, diamond):
        width = self.__overlap([self.x_min, self.x_max], [diamond.x_min, diamond.x_max])
        height = self.__overlap([self.y_min, self.y_max], [diamond.y_min, diamond.y_max])
        return width * height

    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    @property
    def area(self):
        return self.wMax * self.hMax

    @property
    def coords_centroid(self):
        return np.array([self.cx, self.cy, self.wTop, self.hLeft, self.wBottom, self.hRight])

    @coords_centroid.setter
    def coords_centroid(self, coords):
        self.cx, self.cy, self.wTop, self.hLeft, self.wBottom, self.hRight = coords

    @property
    def cx(self):
        return self._x

    @cx.setter
    def cx(self, x):
        self._x = x

    @property
    def cy(self):
        return self._y

    @property
    def wMax(self):
        return max(self.wTop, self.wBottom)

    @property
    def hMax(self):
        return max(self.hLeft, self.hRight)

    @property
    def wMin(self):
        return min(self.wTop, self.wBottom)

    @property
    def hMin(self):
        return min(self.hLeft, self.hRight)

    @cy.setter
    def cy(self, y):
        self._y = y

    @property
    def x_min(self):
        return self.cx - self.wMax / 2

    @property
    def x_max(self):
        return self.cx + self.wMax / 2

    @property
    def y_min(self):
        return self.cy - self.hMax / 2

    @property
    def y_max(self):
        return self.cy + self.hMax / 2


    def __repr__(self):
        return '[({0:.2f},{1:.2f}) --> ({2:.2f},{3:.2f}): ({4:.2f})]'.format(self.x_min, self.y_min, self.x_max,
                                                                             self.y_max, self.class_conf)
