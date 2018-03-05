import copy

import numpy as np
from labels.GateLabel import GateLabel
from labels.ObjectLabel import ObjectLabel

from src.python.utils.labels.ImgLabel import ImgLabel


def resize_label(label: ImgLabel, img_shape=None, shape: tuple = None, scale_x=1.0, scale_y=1.0):
    label_resized = copy.deepcopy(label)
    for i, obj in enumerate(label_resized.objects):
        if obj is None: continue
        obj_resized = resize_label_bb(obj, img_shape, shape, scale_x, scale_y)

        if isinstance(obj_resized, GateLabel):
            obj_resized = resize_label_gate(obj_resized, img_shape, shape, scale_x, scale_y)

        label_resized.objects[i] = obj_resized
    return label_resized


def resize_label_bb(obj: ObjectLabel, img_shape, shape: tuple = None, scale_x=1.0, scale_y=1.0):
    obj_resized = copy.deepcopy(obj)

    if shape is not None:
        scale_y = (shape[0] / img_shape[0])
        scale_x = (shape[1] / img_shape[1])

    obj_resized.y_min *= scale_y
    obj_resized.y_max *= scale_y
    obj_resized.x_min *= scale_x
    obj_resized.x_max *= scale_x

    return obj_resized


def resize_label_gate(obj: GateLabel, img_shape, shape: tuple = None, scale_x=1.0, scale_y=1.0):
    obj_resized = copy.deepcopy(obj)

    if shape is not None:
        scale_y = (shape[0] / img_shape[0])
        scale_x = (shape[1] / img_shape[1])

    scale = np.array([scale_x, scale_y])

    obj_resized.gate_corners.center = np.round(np.multiply(obj_resized.gate_corners.center, scale)).astype(int)
    obj_resized.gate_corners.top_right = np.round(np.multiply(obj_resized.gate_corners.top_right, scale)).astype(int)
    obj_resized.gate_corners.top_left = np.round(np.multiply(obj_resized.gate_corners.top_left, scale)).astype(int)
    obj_resized.gate_corners.bottom_right = np.round(np.multiply(obj_resized.gate_corners.bottom_right, scale)).astype(
        int)
    obj_resized.gate_corners.bottom_left = np.round(np.multiply(obj_resized.gate_corners.bottom_left, scale)).astype(
        int)

    return obj_resized
