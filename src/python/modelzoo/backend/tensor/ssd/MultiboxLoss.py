import keras.backend as K
import tensorflow as tf

from modelzoo.backend.tensor.metrics.Loss import Loss


class MultiboxLoss(Loss):
    def __init__(self, batch_size=8, n_negatives_min=0, negative_positive_ratio=3, loc_class_error_weight=1.0):
        self.alpha = loc_class_error_weight
        self.batch_size = batch_size
        self.n_neg_min = n_negatives_min
        self.neg_pos_ratio = negative_positive_ratio

    def localization_loss(self, y_true, y_pred):
        """
        Calculates the localization loss as the smooth l1_loss between the
        true and predicted coordinates. The loss is weighted with w_positives
        as we only account for the loss for actual predictions.
        :param y_true: tensor(#batch,#boxes,#classes+5) containing the true labels
        :param y_pred: tensor(#batch,#boxes,#classes+5) containing the network predictions
        :return: tensor(#batch,) localization loss per batch
        """
        with K.name_scope('LocalizationLoss'):
            w_positives = K.max(y_true[:, :, 1:-4], -1)
            n_positive = K.sum(w_positives)
            loc_loss_total = K.cast(
                self.smooth_l1_loss(y_true[:, :, -4:], y_pred[:, :, -4:]), K.floatx())

            loc_loss_sum = K.sum(loc_loss_total * w_positives, axis=-1)

            loc_loss_avg = loc_loss_sum / K.maximum(1.0, n_positive)

            return self.alpha * loc_loss_avg

    # noinspection PyMethodMayBeStatic
    def conf_loss_positives(self, y_true, y_pred):
        """
        Calculates the confidence loss for the positive boxes as the softmax -
        crossentropy loss between the true and predicted class confidences.
        The loss is weighted with w_positives as we only account for the loss for positive boxes.
        :param y_true: tensor(#batch,#boxes,#classes+5) containing the true labels
        :param y_pred: tensor(#batch,#boxes,#classes+5) containing the network predictions
        :return: tensor(#batch,) positive confidence loss per batch
        """
        with K.name_scope('PositiveClassificationLoss'):
            w_positives = K.max(y_true[:, :, 1:-4], -1)
            n_positive = K.sum(w_positives)
            class_loss_total = K.categorical_crossentropy(target=y_true[:, :, :-4],
                                                          output=y_pred[:, :, :-4], from_logits=True)

            pos_class_loss = class_loss_total * w_positives
            pos_class_loss_sum = K.sum(pos_class_loss, axis=-1)
            pos_class_loss_avg = pos_class_loss_sum / K.maximum(1.0, n_positive)
            return pos_class_loss_avg

    def conf_loss_negatives(self, y_true, y_pred):
        """
        Calculates the confidence loss for the negative boxes as the softmax -
        crossentropy loss between the true and predicted class confidences.
        (Hard Negative Mining)
        To even out between positive and negative boxes we only calculate the loss
        between the boxes with the highest confidence losses, such that the ratio
        between positive and negative boxes is at most 1:self.neg_pos_ratio
        :param y_true: tensor(#batch,#boxes,#classes+5) containing the true labels
        :param y_pred: tensor(#batch,#boxes,#classes+5) containing the network predictions
        :return: tensor(#batch,) negative confidence loss per batch
        """
        with K.name_scope("NegativeClassificationLoss"):
            w_negatives = y_true[:, :, 0]
            n_negatives = K.sum(w_negatives, axis=-1, keepdims=True)

            w_positives = K.max(y_true[:, :, 1:-4], -1)
            n_positive = K.sum(w_positives)

            class_loss_total = K.categorical_crossentropy(target=y_true[:, :, :-4],
                                                          output=y_pred[:, :, :-4], from_logits=True)

            # We try to take the neg_pos_ration, however at least n_neg_min and at most all boxes
            neg_class_loss = class_loss_total * w_negatives
            n_negative_keep = K.maximum(tf.to_int32(self.n_neg_min),
                                        K.minimum(self.neg_pos_ratio * tf.to_int32(n_positive),
                                                  tf.to_int32(tf.reshape(n_negatives, [-1]))))

            # We get the boxes with the highest confidence loss for each batch
            # top k fails if n negative is zero so we always get one more and sum only the first
            top_losses, _ = tf.nn.top_k(neg_class_loss, K.max(n_negative_keep) + 1, True)
            top_neg_class_loss = []
            for i in range(self.batch_size):
                top_losses_sum = K.sum(top_losses[i, :n_negative_keep[i]])
                top_neg_class_loss_i = K.expand_dims(top_losses_sum, 0)
                top_neg_class_loss.append(top_neg_class_loss_i)

            top_neg_class_loss = K.concatenate(top_neg_class_loss, 0)

            neg_class_loss_sum_average = top_neg_class_loss / K.maximum(1.0, n_positive)

            return neg_class_loss_sum_average

    def compute(self, y_true, y_pred):
        """
        Compute the loss of the ssd model prediction against the ground truth.
        """
        loc_loss = self.localization_loss(y_true, y_pred)
        pos_class_loss = self.conf_loss_positives(y_true, y_pred)

        neg_class_loss = self.conf_loss_negatives(y_true, y_pred)

        class_loss = pos_class_loss + neg_class_loss

        total_loss = class_loss + loc_loss

        return total_loss

    @staticmethod
    def smooth_l1_loss(y_true, y_pred):
        """
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.

        Returns:
            The smooth L1 loss, a nD-1 tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        """
        absolute_loss = K.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = K.switch(K.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return K.sum(l1_loss, axis=-1)
