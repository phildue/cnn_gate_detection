from modelzoo.backend.tensor.metrics.Loss import Loss
import keras.backend as K


class CropAnchorLoss(Loss):
    def compute(self, y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred)
