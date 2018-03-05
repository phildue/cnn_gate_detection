import keras.backend as K
import numpy as np

from src.python.utils import BoundingBox


def iou_np(box_a, box_b):
    """
    Implements the calculation of the intersection-over-union between each box in a and b as matrix operation.
    The tensors are reshaped to
    (len_a, 2) -> (len_a, 1, 2) -> (len_a,len_b,2)
    (len_b, 2) -> (1, len_b, 2) -> (len_a,len_b,2)
    Then the corners can be calculated between each box by matrix operations.
    :param box_a: tensor(len_a,4) containing box coordinates in min-max format
    :param box_b: tensor(len_b,4) containing box coordinates in min-max format
    :return: tensor(len_a,len_b) containing iou between each box
    """

    len_a = box_a.shape[0]
    len_b = box_b.shape[0]

    box_a_reshape = np.reshape(box_a[:, 2:], (len_a, 1, 2))
    box_b_reshape = np.reshape(box_b[:, 2:], (1, len_b, 2))

    box_a_reshape3d = np.tile(box_a_reshape, (1, len_b, 1))
    box_b_reshape3d = np.tile(box_b_reshape, (len_a, 1, 1))
    max_xy = np.min([box_a_reshape3d, box_b_reshape3d], axis=0)

    box_a_reshape = np.reshape(box_a[:, :2], (len_a, 1, 2))
    box_b_reshape = np.reshape(box_b[:, :2], (1, len_b, 2))

    box_a_reshape3d = np.tile(box_a_reshape, (1, len_b, 1))
    box_b_reshape3d = np.tile(box_b_reshape, (len_a, 1, 1))
    min_xy = np.max([box_a_reshape3d, box_b_reshape3d], axis=0)

    width_height = (max_xy - min_xy)
    width_height[width_height < 0] = 0
    area_intersect = width_height[:, :, 0] * width_height[:, :, 1]

    area_a = np.tile(np.reshape((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), (len_a, 1)),
                     (1, len_b))
    area_b = np.tile(np.reshape((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), (1, len_b)),
                     (len_a, 1))

    union = area_a + area_b - area_intersect

    return area_intersect / union


def iou_k(box_a, box_b):
    """
    Implements the calculation of the intersection-over-union between each box in a and b as matrix operation.
    The tensors are reshaped to
    (#batch,len_a, 2) -> (#batch,len_a, 1, 2) -> (#batch,len_a,len_b,2)
    (#batch,len_b, 2) -> (#batch,1, len_b, 2) -> (#batch,len_a,len_b,2)
    Then the corners can be calculated between each box by matrix operations.
    :param box_a: tensor(#batch,len_a,4) containing box coordinates in min-max format
    :param box_b: tensor(#batch,len_b,4) containing box coordinates in min-max format
    :return: tensor(#batch,len_a,len_b) containing iou between each box
    """
    len_a = K.shape(box_a)[1]
    len_b = K.shape(box_b)[1]

    box_a_expand = K.expand_dims(box_a[:, :, 2:], 2)
    box_b_expand = K.expand_dims(box_b[:, :, 2:], 1)

    box_a_reshape3d = K.tile(box_a_expand, [1, 1, len_b, 1])
    box_b_reshape3d = K.tile(box_b_expand, [1, len_a, 1, 1])
    max_xy = K.min([box_a_reshape3d, box_b_reshape3d], axis=0)

    box_a_expand = K.expand_dims(box_a[:, :, :2], 2)
    box_b_expand = K.expand_dims(box_b[:, :, :2], 1)

    box_a_reshape3d = K.tile(box_a_expand, [1, 1, len_b, 1])
    box_b_reshape3d = K.tile(box_b_expand, [1, len_a, 1, 1])
    min_xy = K.max([box_a_reshape3d, box_b_reshape3d], axis=0)

    width_height = (max_xy - min_xy)
    width_height = K.maximum(width_height, 0)

    area_intersect = width_height[:, :, :, 0] * width_height[:, :, :, 1]

    area_a = K.tile(K.reshape((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]), (-1, len_a, 1)),
                    (1, 1, len_b))
    area_b = K.tile(K.reshape((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]), (-1, 1, len_b)),
                    (1, len_a, 1))

    union = area_a + area_b - area_intersect

    iou = area_intersect / union

    iou = K.cast(iou, K.floatx())

    return iou


def non_max_suppression(boxes: [BoundingBox], iou_thresh=0.4, n_max=50):
    coord_t = BoundingBox.to_tensor_minmax(boxes)
    confs = [b.c for b in boxes]
    conf_t = np.array(confs).flatten()

    idx = K.get_session().run(non_max_suppression_tf(K.constant(coord_t), K.constant(conf_t), iou_thresh, n_max))

    return [b for i, b in enumerate(boxes) if i in idx]


def non_max_suppression_tf(boxes_pred_t, class_conf_t, iou_thresh, n_max=50):
    return K.tf.image.non_max_suppression(boxes_pred_t, class_conf_t, n_max, iou_thresh, 'NonMaxSuppression')
