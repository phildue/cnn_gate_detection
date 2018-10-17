import keras.backend as K

from modelzoo.metrics.Loss import Loss


class PolygonLoss(Loss):
    def compute(self, y_true, y_pred):
        return K.mean(K.pow(y_true - y_pred, 2), -1)
