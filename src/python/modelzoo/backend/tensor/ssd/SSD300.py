import keras.backend as K
from keras import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Reshape, Lambda

from modelzoo.backend.tensor.metrics.Loss import Loss
from modelzoo.backend.tensor.ssd.ConcatMeta import ConcatMeta
from modelzoo.backend.tensor.ssd.L2Normalization import L2Normalization
from modelzoo.backend.tensor.ssd.SSDNet import SSDNet


class SSD300(SSDNet):
    @property
    def anchors(self):
        return self._anchors

    def backend(self):
        return self._model

    def __init__(self, variances, img_shape: (int, int, int), loss: Loss, scales, aspect_ratios,
                 weight_file=None, n_classes=20, n_boxes=None):
        super().__init__(img_shape, variances, scales, aspect_ratios, loss)
        self.n_classes = n_classes
        self.n_boxes_conv4 = n_boxes['conv4']
        self.n_boxes_fc7 = n_boxes['fc7']
        self.n_boxes_conv8 = n_boxes['conv8']
        self.n_boxes_conv9 = n_boxes['conv9']
        self.n_boxes_conv10 = n_boxes['conv10']
        self.n_boxes_conv11 = n_boxes['conv11']
        self._model, self._anchors = self.build_model()

        if weight_file is not None:
            self._model.load_weights(weight_file, by_name=True)

    def build_model(self):
        # Input image format
        img_height, img_width, img_channels = self.img_shape[0], self.img_shape[1], self.img_shape[2]

        x = Input(shape=(img_height, img_width, img_channels))
        normed = Lambda(lambda z: z / 127.5 - 1.0,  # Convert input feature range to [-1,1]
                        output_shape=(img_height, img_width, img_channels),
                        name='scale')(x)

        with K.name_scope('BaseNetwork'):
            conv4, conv8, conv9, conv10, conv11, fc7 = self._build_base_network(normed)

        with K.name_scope('ClassPredictors'):
            pred_c, predictor_sizes = self._build_class_predictors(conv4, fc7, conv8, conv9, conv10, conv11)

        with K.name_scope('LocalizationPredictors'):
            pred_loc = self._build_loc_predictors(conv4, fc7, conv8, conv9, conv10, conv11)

        predictions = Concatenate(axis=2, name='predictions')([pred_c, pred_loc])

        anchors = self.generate_anchors_t(predictor_sizes)
        meta_t = self._generate_meta_t(anchors)

        netout = ConcatMeta((K.shape(predictions)), meta_t)(predictions)

        return Model(inputs=x, outputs=netout), anchors

    def _build_loc_predictors(self, conv4, fc7, conv8, conv9, conv10, conv11):
        pred4_loc = Conv2D(self.n_boxes_conv4 * 4, (3, 3), padding='same', name='pred4_loc')(
            conv4)
        pred7_loc = Conv2D(self.n_boxes_fc7 * 4, (3, 3), padding='same', name='pred7_loc')(fc7)
        pred8_loc = Conv2D(self.n_boxes_conv8 * 4, (3, 3), padding='same', name='pred8_loc')(conv8)
        pred9_loc = Conv2D(self.n_boxes_conv9 * 4, (3, 3), padding='same', name='pred9_loc')(conv9)
        pred10_loc = Conv2D(self.n_boxes_conv10 * 4, (3, 3), padding='same', name='pred10_loc')(conv10)
        pred11_loc = Conv2D(self.n_boxes_conv11 * 4, (3, 3), padding='same', name='pred11_loc')(conv11)

        pred4_loc_reshape = Reshape((-1, 4), name='pred4_loc_reshape')(
            pred4_loc)
        pred7_loc_reshape = Reshape((-1, 4), name='pred7_loc_reshape')(pred7_loc)
        pred8_loc_reshape = Reshape((-1, 4), name='pred8_loc_reshape')(pred8_loc)
        pred9_loc_reshape = Reshape((-1, 4), name='pred9_loc_reshape')(pred9_loc)
        pred10_loc_reshape = Reshape((-1, 4), name='pred10_loc_reshape')(pred10_loc)
        pred11_loc_reshape = Reshape((-1, 4), name='pred11_loc_reshape')(pred11_loc)

        return Concatenate(axis=1, name='pred_loc')([pred4_loc_reshape,
                                                     pred7_loc_reshape,
                                                     pred8_loc_reshape,
                                                     pred9_loc_reshape,
                                                     pred10_loc_reshape,
                                                     pred11_loc_reshape])

    def _build_class_predictors(self, conv4_3_norm, fc7, conv6_2, conv7_2, conv8_2, conv9_2):
        n_classes = self.n_classes

        pred4_c = Conv2D(self.n_boxes_conv4 * n_classes, (3, 3), padding='same',
                         name='pred4_c')(conv4_3_norm)
        pred7_c = Conv2D(self.n_boxes_fc7 * n_classes, (3, 3), padding='same', name='pred7_c')(fc7)

        pred8_c = Conv2D(self.n_boxes_conv8 * n_classes, (3, 3), padding='same', name='pred8_c')(
            conv6_2)

        pred9_c = Conv2D(self.n_boxes_conv9 * n_classes, (3, 3), padding='same', name='pred9_c')(
            conv7_2)

        pred10_c = Conv2D(self.n_boxes_conv10 * n_classes, (3, 3), padding='same', name='pred10_c')(
            conv8_2)

        pred11_c = Conv2D(self.n_boxes_conv11 * n_classes, (3, 3), padding='same', name='pred11_c')(
            conv9_2)

        pred4_c_reshape = Reshape((-1, n_classes), name='pred4_c_reshape')(
            pred4_c)
        pred7_c_reshape = Reshape((-1, n_classes), name='pred7_c_reshape')(pred7_c)
        pred8_c_reshape = Reshape((-1, n_classes), name='pred8_c_reshape')(pred8_c)
        pred9_c_reshape = Reshape((-1, n_classes), name='pred9_c_reshape')(pred9_c)
        pred10_c_reshape = Reshape((-1, n_classes), name='pred10_c_reshape')(pred10_c)
        pred11_c_reshape = Reshape((-1, n_classes), name='pred11_c_reshape')(pred11_c)

        pred_c = Concatenate(axis=1, name='pred_c')([pred4_c_reshape,
                                                     pred7_c_reshape,
                                                     pred8_c_reshape,
                                                     pred9_c_reshape,
                                                     pred10_c_reshape,
                                                     pred11_c_reshape])

        predictor_sizes = K.np.array([pred4_c._keras_shape[1:3],
                                      pred7_c._keras_shape[1:3],
                                      pred8_c._keras_shape[1:3],
                                      pred9_c._keras_shape[1:3],
                                      pred10_c._keras_shape[1:3],
                                      pred11_c._keras_shape[1:3]])

        return pred_c, predictor_sizes

    @staticmethod
    def _build_base_network(feed_in):
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(feed_in)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool4')(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5)

        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5)

        fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)

        fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)

        conv8 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv8_1')(fc7)
        conv8 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(conv8)

        conv9 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8)
        conv9 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv9_2')(conv9)

        conv10 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv10_1')(conv9)
        conv10 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv10_2')(conv10)

        conv11 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv11_1')(conv10)
        conv11 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv11_2')(conv11)

        conv4_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4)

        return conv4_norm, conv8, conv9, conv10, conv11, fc7
