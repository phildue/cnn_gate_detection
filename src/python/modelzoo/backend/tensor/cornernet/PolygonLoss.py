from modelzoo.backend.tensor.metrics.Loss import Loss
import keras.backend as K


class PolygonLoss(Loss):
    def compute(self, y_true, y_pred):
        return K.mean(K.pow(y_true - y_pred, 2), -1)
