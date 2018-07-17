import numpy as np

from utils.imageprocessing.Imageprocessing import get_bounding_box
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel


def centroid_to_minmax(coord_t):
    coord_t_minmax = np.zeros(coord_t.shape)
    coord_t_minmax[:, 0] = coord_t[:, 0] - coord_t[:, 2] / 2
    coord_t_minmax[:, 1] = coord_t[:, 1] - coord_t[:, 3] / 2
    coord_t_minmax[:, 2] = coord_t[:, 0] + coord_t[:, 2] / 2
    coord_t_minmax[:, 3] = coord_t[:, 1] + coord_t[:, 3] / 2
    return coord_t_minmax


def minmax_to_centroid(coord_t):
    coord_t_centroid = np.zeros(coord_t.shape)
    coord_t_centroid[:, 2] = coord_t[:, 2] - coord_t[:, 0]
    coord_t_centroid[:, 3] = coord_t[:, 3] - coord_t[:, 1]
    coord_t_centroid[:, 0] = coord_t[:, 0] + coord_t_centroid[:, 2] / 2
    coord_t_centroid[:, 1] = coord_t[:, 1] + coord_t_centroid[:, 3] / 2
    return coord_t_centroid


class BoundingBox:
    @staticmethod
    def from_label(label: ImgLabel):
        boxes = []
        for l in label.objects:
            if l is None: continue
            if isinstance(l, GateLabel):
                p1, p2 = get_bounding_box(l.gate_corners)
                l = ObjectLabel('gate', np.array([p1, p2]))
            b = BoundingBox(len(ObjectLabel.classes))
            b.w = l.x_max - l.x_min
            b.h = l.y_max - l.y_min
            b.cx = l.x_min + b.w / 2
            b.cy = l.y_min + b.h / 2
            b.c = l.confidence
            b.probs[ObjectLabel.name_to_id(l.class_name)] = l.confidence
            boxes.append(b)
        return boxes

    @staticmethod
    def to_label(boxes) -> ImgLabel:
        """
        Convert boxes to BoundingBoxLabels
        :param boxes: objects of type box
        :return: bounding box labels
        """
        box_labels = []
        for box in boxes:
            try:
                xmin = int((box.cx - box.w / 2))
                xmax = int((box.cx + box.w / 2))
                ymin = int((box.cy - box.h / 2))
                ymax = int((box.cy + box.h / 2))
            except ValueError:
                xmin = 0
                xmax = 0
                ymin = 0
                ymax = 0
            box_labels.append(ObjectLabel(ObjectLabel.id_to_name(box.prediction),
                                          np.array([[xmin, ymin],
                                                    [xmax, ymax]]), box.class_conf))
        return ImgLabel(box_labels)

    @staticmethod
    def to_tensor_minmax(boxes):
        return np.concatenate([np.expand_dims(b.coords_minmax, 0) for b in boxes], 0)

    @staticmethod
    def to_tensor_centroid(boxes):
        return np.concatenate([np.expand_dims(b.coords_centroid, 0) for b in boxes], 0)

    @staticmethod
    def to_class_tensor(boxes):
        return np.concatenate([np.expand_dims(b.probs.T, 0) for b in boxes], 0)

    @staticmethod
    def from_tensor_centroid(class_t, coord_t, conf_t=None):
        coord_t_minmax = centroid_to_minmax(coord_t)
        return BoundingBox.from_tensor_minmax(class_t, coord_t_minmax, conf_t)

    @staticmethod
    def from_tensor_minmax(class_t, coord_t, conf_t=None):
        n_boxes = class_t.shape[0]
        n_classes = class_t.shape[1]
        boxes = []
        for i in range(n_boxes):
            b = BoundingBox(n_classes)
            b.coords_minmax = coord_t[i]
            b.c = conf_t[i] if conf_t is not None else np.max(class_t[i])
            b.probs = class_t[i]
            boxes.append(b)
        return boxes

    def __init__(self, class_num):
        self._x, self._y, self._w, self._h, self._c = 0., 0., 0., 0., 0.
        self.probs = np.zeros((class_num,))

    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w * self.h + box.w * box.h - intersection
        if union == 0:
            return 1
        else:
            return intersection / union

    @property
    def prediction(self):
        return np.argmax(self.probs)

    @property
    def class_conf(self):
        return np.max(self.probs)

    def intersect(self, box):
        width = self.__overlap([self.x_min, self.x_max], [box.x_min, box.x_max])
        height = self.__overlap([self.y_min, self.y_max], [box.y_min, box.y_max])
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
        return self.w * self.h

    @property
    def coords_minmax(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])

    @coords_minmax.setter
    def coords_minmax(self, coords):
        x_min, y_min, x_max, y_max = coords

        self.h = y_max - y_min
        self.w = x_max - x_min
        self.x_min = x_min
        self.y_min = y_min

    @property
    def coords_centroid(self):
        return np.array([self.cx, self.cy, self.w, self.h])

    @coords_centroid.setter
    def coords_centroid(self, coords):
        self.cx, self.cy, self.w, self.h = coords

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
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @cy.setter
    def cy(self, y):
        self._y = y

    @w.setter
    def w(self, w):
        self._w = w

    @h.setter
    def h(self, h):
        self._h = h

    @property
    def x_min(self):
        return self.cx - self.w / 2

    @x_min.setter
    def x_min(self, x):
        self.cx = x + self.w / 2

    @property
    def x_max(self):
        return self.cx + self.w / 2

    @x_max.setter
    def x_max(self, x):
        self.cx = x - self.w / 2

    @property
    def y_min(self):
        return self.cy - self.h / 2

    @property
    def y_max(self):
        return self.cy + self.h / 2

    @y_max.setter
    def y_max(self, y):
        self.cy = y - self.h / 2

    @y_min.setter
    def y_min(self, y):
        self.cy = y + self.h / 2

    def __repr__(self):
        return '[({0:.2f},{1:.2f}) --> ({2:.2f},{3:.2f}): ({4:.2f})]'.format(self.x_min, self.y_min, self.x_max,
                                                                             self.y_max, self.class_conf)
