import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, Reshape, Concatenate
from keras.optimizers import Adam

from modelzoo.backend.tensor.ConcatMeta import ConcatMeta
from modelzoo.backend.tensor.layers import create_layer
from modelzoo.backend.tensor.yolo.Netout import Netout
from modelzoo.models.Net import Net
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from modelzoo.models.yolo.YoloEncoder import YoloEncoder


class YoloBase(Net):

    def compile(self, params=None, metrics=None):
        # default_sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9)
        if params is not None:
            self._params = params

        optimizer = Adam(self._params['lr'], self._params['beta_1'], self._params['beta_2'], self._params['epsilon'],
                         self._params['decay'])

        self._model.compile(
            loss=self.loss.compute,
            optimizer=optimizer,
            metrics=metrics
        )

    @property
    def backend(self):
        return self._model

    @backend.setter
    def backend(self, model):
        self._model = model

    @property
    def train_params(self):
        return self._params

    def predict(self, sample):
        return self._model.predict(sample)

    def __init__(self, architecture,
                 anchors,
                 loss,
                 n_classes,
                 img_shape,
                 n_boxes,
                 input_channels=3,
                 weight_file=None):

        self.n_classes = n_classes
        self.loss = loss
        self.norm = img_shape
        self.n_boxes = n_boxes
        self.anchors = anchors
        self._params = {'optimizer': 'adam',
                        'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-08,
                        'decay': 0.0005}

        h, w = img_shape
        netin = Input((h, w, input_channels))

        net = netin
        self.grid = []
        prediction_layer_i = 0
        predictions = []
        layers = []
        n_outputs_per_box = 4 + 1 + self.n_classes  # cx,cy,w,h,object,class
        for i, config in enumerate(architecture):
            if 'predict' in config['name']:
                with K.name_scope('predict{}'.format(prediction_layer_i)):
                    inference = Conv2D(n_boxes[prediction_layer_i] * n_outputs_per_box, kernel_size=(1, 1),
                                       strides=(1, 1), name='predictor{}'.format(prediction_layer_i))(
                        net)
                    prediction_layer_i += 1
                    reshape = Reshape((-1, n_outputs_per_box))(inference)
                    prediction = Netout(self.n_classes)(reshape)
                    predictions.append(prediction)
                    layers.append(inference)
                    grid = K.int_shape(net)[-3], K.int_shape(net)[-2]
                    self.grid.append(grid)
            elif 'route' in config['name']:
                if len(config['index']) > 1:
                    net = Concatenate()([layers[i] for i in config['index']])
                else:
                    net = layers[config['index'][0]]
                layers.append(net)
            else:
                with K.name_scope('layer' + str(i)):
                    net = create_layer(net, config)
                    layers.append(net)
        if len(predictions) > 1:
            predictions = Concatenate(-2)(predictions)
        else:
            predictions = predictions[0]

        meta_t = K.constant(GateNetEncoder.generate_anchors(self.norm, self.grid, self.anchors, 4),
                            K.tf.float32)

        netout = ConcatMeta(meta_t)(predictions)
        model = Model(netin, netout)

        if weight_file is not None:
            model.load_weights(weight_file)

        self._model = model
