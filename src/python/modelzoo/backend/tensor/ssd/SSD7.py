import keras.backend as K
from keras import Model, Input
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, ELU, Concatenate, Reshape, Lambda

from modelzoo.backend.tensor.metrics.Loss import Loss
from modelzoo.backend.tensor.ssd.ConcatMeta import ConcatMeta
from modelzoo.backend.tensor.ssd.SSDNet import SSDNet


class SSD7(SSDNet):
    @property
    def anchors(self):
        return self._anchors

    @property
    def backend(self):
        return self._model

    @backend.setter
    def backend(self, model):
        self._model = model

    def __init__(self, img_shape,
                 variances,
                 scales,
                 aspect_ratios,
                 loss: Loss,
                 weight_file=None,
                 n_classes=20,
                 n_boxes=None):
        super().__init__(img_shape, variances, scales, aspect_ratios, loss)
        self.n_classes = n_classes
        self.n_boxes_conv4 = n_boxes['conv4']
        self.n_boxes_conv5 = n_boxes['conv5']
        self.n_boxes_conv6 = n_boxes['conv6']
        self.n_boxes_conv7 = n_boxes['conv7']

        self._model, self._anchors = self.build_model()

        if weight_file is not None:
            self._model.load_weights(weight_file, by_name=True)

    def build_model(self):
        # Input image format
        img_height, img_width, img_channels = self.img_shape[0], self.img_shape[1], self.img_shape[2]

        # Design the actual network
        x = Input(shape=(img_height, img_width, img_channels))
        normed = Lambda(lambda z: z / 127.5 - 1.,  # Convert input feature range to [-1,1]
                        output_shape=(img_height, img_width, img_channels),
                        name='scale')(x)

        with K.name_scope('BaseNetwork'):
            conv4, conv5, conv6, conv7 = self._build_base_network(normed)

        with K.name_scope('ClassPredictors'):
            pred_c, predictor_sizes = self._build_class_predictors(conv4, conv5, conv6, conv7)

        with K.name_scope('LocalizationPredictors'):
            pred_loc = self._build_loc_predictors(conv4, conv5, conv6, conv7)

        predictions = Concatenate(axis=2, name='predictions')([pred_c, pred_loc])

        anchors = self.generate_anchors_t(predictor_sizes)
        meta_t = self._generate_meta_t(anchors)

        netout = ConcatMeta((K.shape(predictions)), meta_t)(predictions)

        return Model(inputs=x, outputs=netout), anchors

    def _build_loc_predictors(self, conv4, conv5, conv6, conv7):
        boxes4 = Conv2D(self.n_boxes_conv4 * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes4')(conv4)
        boxes5 = Conv2D(self.n_boxes_conv5 * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes5')(conv5)
        boxes6 = Conv2D(self.n_boxes_conv6 * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes6')(conv6)
        boxes7 = Conv2D(self.n_boxes_conv7 * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes7')(conv7)

        boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
        boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
        boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
        boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)

        return Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped,
                                                         boxes5_reshaped,
                                                         boxes6_reshaped,
                                                         boxes7_reshaped])

    def _build_class_predictors(self, conv4, conv5, conv6, conv7):
        classes4 = Conv2D(self.n_boxes_conv4 * self.n_classes, (3, 3), strides=(1, 1), padding="valid",
                          name='classes4')(
            conv4)
        classes5 = Conv2D(self.n_boxes_conv5 * self.n_classes, (3, 3), strides=(1, 1), padding="valid",
                          name='classes5')(
            conv5)
        classes6 = Conv2D(self.n_boxes_conv6 * self.n_classes, (3, 3), strides=(1, 1), padding="valid",
                          name='classes6')(
            conv6)
        classes7 = Conv2D(self.n_boxes_conv7 * self.n_classes, (3, 3), strides=(1, 1), padding="valid",
                          name='classes7')(
            conv7)

        classes4_reshaped = Reshape((-1, self.n_classes), name='classes4_reshape')(classes4)
        classes5_reshaped = Reshape((-1, self.n_classes), name='classes5_reshape')(classes5)
        classes6_reshaped = Reshape((-1, self.n_classes), name='classes6_reshape')(classes6)
        classes7_reshaped = Reshape((-1, self.n_classes), name='classes7_reshape')(classes7)

        classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                                     classes5_reshaped,
                                                                     classes6_reshaped,
                                                                     classes7_reshaped])
        predictor_sizes = K.np.array([classes4._keras_shape[1:3],
                                      classes5._keras_shape[1:3],
                                      classes6._keras_shape[1:3],
                                      classes7._keras_shape[1:3]])

        return classes_concat, predictor_sizes

    @staticmethod
    def _build_base_network(feed_in):
        conv1 = Conv2D(32, (5, 5), name='conv1', strides=(1, 1), padding="same")(feed_in)
        conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(
            conv1)
        conv1 = ELU(name='elu1')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

        conv2 = Conv2D(48, (3, 3), name='conv2', strides=(1, 1), padding="same")(pool1)
        conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
        conv2 = ELU(name='elu2')(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

        conv3 = Conv2D(64, (3, 3), name='conv3', strides=(1, 1), padding="same")(pool2)
        conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
        conv3 = ELU(name='elu3')(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

        conv4 = Conv2D(64, (3, 3), name='conv4', strides=(1, 1), padding="same")(pool3)
        conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
        conv4 = ELU(name='elu4')(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

        conv5 = Conv2D(48, (3, 3), name='conv5', strides=(1, 1), padding="same")(pool4)
        conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
        conv5 = ELU(name='elu5')(conv5)

        pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

        conv6 = Conv2D(48, (3, 3), name='conv6', strides=(1, 1), padding="same")(pool5)
        conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
        conv6 = ELU(name='elu6')(conv6)

        pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

        conv7 = Conv2D(32, (3, 3), name='conv7', strides=(1, 1), padding="same")(pool6)
        conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
        conv7 = ELU(name='elu7')(conv7)

        return conv4, conv5, conv6, conv7
