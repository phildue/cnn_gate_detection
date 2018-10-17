import keras.backend as K

from modelzoo.metrics.Loss import Loss


class GateDetectionLoss(Loss):
    def __init__(self, n_boxes=5, n_polygon=4, weight_loc=5.0, weight_noobj=0.5,
                 weight_conf=5.0, weight_prob=1.0):
        self.scale_prob = weight_prob
        self.scale_noob = weight_noobj
        self.scale_obj = weight_conf
        self.scale_coor = weight_loc
        self.n_polygon = n_polygon
        self.n_boxes = n_boxes

    def loss(self, y_true, y_pred):
        y_true_k = K.constant(y_true, name="y_true")
        y_pred_k = K.constant(y_pred, name="netout")

        return K.get_session().run(self.compute(y_true=y_true_k, y_pred=y_pred_k))

    def compute(self, y_true, y_pred):
        """
        Loss function for GateNet.
        :param y_true: y as fed for learning
        :param y_pred: raw network output
        :return: loss
        """
        # y_pred = K.print_tensor(y_pred, "Y_Pred")
        # y_true = K.print_tensor(y_true, "Y_True")
        loc_loss = self.localization_loss(y_true, y_pred)

        conf_loss = self.confidence_loss(y_true, y_pred)

        return loc_loss + conf_loss

    def localization_loss(self, y_true, y_pred):
        positives = y_true[:, :, 0]

        w_pos = self.scale_coor * K.stack(4 * [positives], -1)
        coord_true = y_true[:, :, 1:5]
        coord_pred = y_pred[:, :, 1:5]

        xy_true = coord_true[:, :, :2]
        xy_pred = coord_pred[:, :, :2]
        xy_loss = K.pow(xy_true - xy_pred, 2)

        wh_true = coord_true[:, :, 2:]
        wh_pred = coord_pred[:, :, 2:]
        wh_loss = K.pow(K.sqrt(wh_true) - K.sqrt(wh_pred), 2)

        loc_loss = K.concatenate([xy_loss, wh_loss], -1) * w_pos

        loc_loss_sum = .5 * K.sum(K.sum(loc_loss, -1), -1)

        # loc_loss_sum = K.print_tensor(loc_loss_sum,'Loc Loss=')

        return loc_loss_sum

    def confidence_loss(self, y_true, y_pred):
        positives = y_true[:, :, 0:1]

        weight = self.scale_noob * (1. - positives) + self.scale_obj * positives

        conf_pred = y_pred[:, :, 0:1]
        conf_true = y_true[:, :, 0:1]

        conf_loss = K.binary_crossentropy(target=conf_true, output=conf_pred, from_logits=True) * weight

        conf_loss_total = .5 * K.sum(K.sum(conf_loss, -1), -1)

        # conf_loss_total = K.print_tensor(conf_loss_total,'Conf Loss=')

        return conf_loss_total
