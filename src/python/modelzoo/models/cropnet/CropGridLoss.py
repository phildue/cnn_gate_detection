import keras.backend as K

from modelzoo.metrics.Loss import Loss


class CropGridLoss(Loss):
    def compute(self, y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred)
