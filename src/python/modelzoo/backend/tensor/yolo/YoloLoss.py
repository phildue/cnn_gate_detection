import keras.backend as K

from src.python.modelzoo.backend.tensor import Loss


class YoloLoss(Loss):
    def __init__(self, grid=(13, 13), n_boxes=5, n_classes=20, verbose=False, weight_loc=5.0, weight_noobj=0.5,
                 weight_conf=5.0, weight_prob=1.0):
        self.scale_prob = weight_prob
        self.scale_noob = weight_noobj
        self.scale_obj = weight_conf
        self.scale_coor = weight_loc
        self.verbose = verbose
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.grid = grid

    @staticmethod
    def _get_iou(box1, box2):
        """
        calculates intersection over union
        :param box1: tensor with [x,y,w,h]
        :param box2: tensor with [x,y,w,h]
        :return: iou
        """
        box1_ul, box1_br, box1_area = YoloLoss._get_box_limits(box1[:, :, :, :, :2], box1[:, :, :, :, 2:])
        box2_ul, box2_br, box2_area = YoloLoss._get_box_limits(box2[:, :, :, :, :2], box2[:, :, :, :, 2:])
        intersect = YoloLoss.get_intersect(box1_ul, box1_br, box2_ul, box2_br)
        return K.tf.truediv(intersect, box1_area + box2_area - intersect)

    @staticmethod
    def _get_box_limits(xy, wh):
        """
        Calculates the box corners and area.
        :param y: label
        :return: upper left corner, bottom right corner, area
        """
        area = wh[:, :, :, :, 0] * wh[:, :, :, :, 1]
        ul = xy - 0.5 * wh
        br = xy + 0.5 * wh

        return ul, br, area

    @staticmethod
    def get_intersect(true_ul, true_br, pred_ul, pred_br):
        """
        Calculates the intersection based on box coordinates.
        :param true_ul: Upper left limit true box
        :param true_br: Lower right limit true box
        :param pred_ul: Upper left limit predicted box
        :param pred_br: Lower right limit predicted box
        :return: intersection area
        """
        intersect_ul = K.maximum(pred_ul, true_ul)
        intersect_br = K.minimum(pred_br, true_br)
        intersect_wh = intersect_br - intersect_ul
        intersect_wh = K.maximum(intersect_wh, 0.0)
        return intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

    def loss(self, y_true, y_pred):
        y_true_k = K.constant(y_true, name="y_true")
        y_pred_k = K.constant(y_pred, name="netout")

        return K.get_session().run(self.compute(y_true=y_true_k, y_pred=y_pred_k))

    def match_true_boxes(self, y_true, y_pred):
        y_true_k = K.constant(y_true, name="y_true")
        y_pred_k = K.constant(y_pred, name="netout")

        return K.get_session().run(self._assign_anchors(y_true_k,
                                                        self._get_iou(
                                                            y_pred_k[:, :, :, :, :4],
                                                            y_true_k[:, :, :, :, :4])))

    def compute(self, y_true, y_pred):
        """
        Loss function for yolo v2.
        :param y_true: y as fed for learning
        :param y_pred: raw network output
        :return: loss
        """
        loc_loss = self.localization_loss(y_true, y_pred)

        conf_loss = self.confidence_loss(y_true, y_pred)

        class_loss = self.class_loss(y_true, y_pred)

        return loc_loss + conf_loss + class_loss

    def localization_loss(self, y_true, y_pred):
        y_true = K.reshape(y_true, [-1, self.grid[0], self.grid[1], self.n_boxes, self.n_classes + 5])
        y_pred = K.reshape(y_pred, [-1, self.grid[0], self.grid[1], self.n_boxes, self.n_classes + 5])

        y_true_assigned = self._assign_anchors(y_true, y_pred)

        positives = y_true_assigned[:, :, :, :, 4]

        weight = self.scale_coor * K.stack(4 * [positives], 4)

        xy_true = y_true_assigned[:, :, :, :, :2]
        xy_pred = y_pred[:, :, :, :, :2]
        xy_loss = K.pow(xy_true - xy_pred, 2)

        wh_true = y_true_assigned[:, :, :, :, 2:4]
        wh_pred = y_pred[:, :, :, :, 2:4]
        wh_loss = K.pow(K.sqrt(wh_true) - K.sqrt(wh_pred), 2)

        loc_loss = K.concatenate([xy_loss, wh_loss], 4) * weight

        loc_loss_sum = .5 * K.sum(K.reshape(loc_loss, (-1, self.grid[0] * self.grid[1] * self.n_boxes * 4)), -1)

        return loc_loss_sum

    def confidence_loss(self, y_true, y_pred):
        y_true = K.reshape(y_true, [-1, self.grid[0], self.grid[1], self.n_boxes, self.n_classes + 5])
        y_pred = K.reshape(y_pred, [-1, self.grid[0], self.grid[1], self.n_boxes, self.n_classes + 5])

        y_true_assigned = self._assign_anchors(y_true, y_pred)

        positives = y_true_assigned[:, :, :, :, 4]

        weight = self.scale_noob * (1. - positives) + self.scale_obj * positives

        conf_pred = y_pred[:, :, :, :, 4]

        conf_true = y_true_assigned[:, :, :, :, 4]

        conf_loss = K.pow(conf_pred - conf_true, 2) * weight

        conf_loss_total = .5 * K.sum(K.reshape(conf_loss, (-1, self.grid[0] * self.grid[1] * self.n_boxes)), -1)

        return conf_loss_total

    def class_loss(self, y_true, y_pred):
        y_true = K.reshape(y_true, [-1, self.grid[0], self.grid[1], self.n_boxes, self.n_classes + 5])
        y_pred = K.reshape(y_pred, [-1, self.grid[0], self.grid[1], self.n_boxes, self.n_classes + 5])

        y_true_assigned = self._assign_anchors(y_true, y_pred)

        positives = y_true_assigned[:, :, :, :, 4]

        weight = self.scale_prob * K.stack(self.n_classes * [positives], 4)

        class_pred = y_pred[:, :, :, :, 5:]
        class_true = y_true_assigned[:, :, :, :, 5:]
        class_loss = K.pow(class_pred - class_true, 2) * weight

        class_loss_sum = .5 * K.sum(
            K.reshape(class_loss, (-1, self.grid[0] * self.grid[1] * self.n_boxes * self.n_classes)), -1)

        return class_loss_sum

    @staticmethod
    def _assign_anchors(y_true, y_pred):
        """
        Adapts the ground truth confidences, according to intersection-over-union. The box with the highest iou
        gets "responsible" for that box.
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: y_true with updated confidences
        """

        coord_true = y_true[:, :, :, :, :4]
        coord_pred = y_pred[:, :, :, :, :4]

        iou = YoloLoss._get_iou(coord_true, coord_pred)

        best_box = K.equal(iou, K.tf.reduce_max(iou, [3], True))
        best_box = K.tf.to_float(best_box)
        true_conf = K.expand_dims(best_box * y_true[:, :, :, :, 4], -1)

        true_prob = y_true[:, :, :, :, 5:]
        true_xy = y_true[:, :, :, :, :2]
        true_wh = y_true[:, :, :, :, 2:4]

        return K.concatenate([true_xy, true_wh, true_conf, true_prob], 4)
