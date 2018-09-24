import copy

import numpy as np

from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel


def resize_label(label: ImgLabel, img_shape=None, shape: tuple = None, scale_x=1.0, scale_y=1.0):
    label_resized = copy.deepcopy(label)
    for i, obj in enumerate(label_resized.objects):
        if obj is None: continue

        if isinstance(obj, GateLabel):
            obj_resized = resize_label_gate(obj, img_shape, shape, scale_x, scale_y)
        elif isinstance(obj, ObjectLabel):
            obj_resized = resize_label_bb(obj, img_shape, shape, scale_x, scale_y)
        else:
            raise ValueError('Unknown Type')

        label_resized.objects[i] = obj_resized
    return label_resized


def resize_label_bb(obj: ObjectLabel, img_shape, shape: tuple = None, scale_x=1.0, scale_y=1.0):
    obj_resized = copy.deepcopy(obj)

    if shape is not None:
        scale_y = (shape[0] / img_shape[0])
        scale_x = (shape[1] / img_shape[1])

    scale = np.array([scale_x, scale_y])
    mat = obj_resized.mat.astype(scale.dtype)
    mat *= scale

    obj_resized.mat = mat

    return obj_resized


def resize_label_gate(obj: GateLabel, img_shape, shape: tuple = None, scale_x=1.0, scale_y=1.0):
    obj_resized = copy.deepcopy(obj)

    if shape is not None:
        scale_y = (shape[0] / img_shape[0])
        scale_x = (shape[1] / img_shape[1])

    scale = np.array([scale_x, scale_y])

    mat = obj_resized.gate_corners.mat.astype(scale.dtype)
    mat *= scale
    obj_resized.gate_corners = GateCorners.from_mat(mat)

    return obj_resized
