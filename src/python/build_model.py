import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, Reshape, Concatenate

from modelzoo.backend.tensor.ConcatMeta import ConcatMeta
from modelzoo.backend.tensor.gatenet.Netout import Netout
from modelzoo.backend.tensor.layers import create_layer
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder


def build_detector(img_shape, architecture, anchors, n_polygon=4):
    h, w, input_channels = img_shape
    n_boxes = [len(a) for a in anchors]

    netin = Input((h, w, input_channels))

    net = netin
    grids = []
    prediction_layer_i = 0
    predictions = []
    layers = []
    for i, config in enumerate(architecture):
        if 'predict' in config['name']:
            with K.name_scope('predict{}'.format(prediction_layer_i)):
                inference = Conv2D(n_boxes[prediction_layer_i] * (n_polygon + 1), kernel_size=(1, 1),
                                   strides=(1, 1), name='predictor{}'.format(prediction_layer_i))(
                    net)
                prediction_layer_i += 1
                reshape = Reshape((-1, n_polygon + 1))(inference)
                prediction = Netout(n_polygon)(reshape)
                predictions.append(prediction)
                layers.append(inference)
                grid = K.int_shape(net)[-3], K.int_shape(net)[-2]
                grids.append(grid)
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

    meta_t = K.constant(GateNetEncoder.generate_anchors((h, w), grids, anchors, n_polygon),
                        K.tf.float32)

    netout = ConcatMeta(meta_t)(predictions)
    model = Model(netin, netout)
    return model


def create_detector(input_channels, architecture, anchors, n_polygon=4):
    n_boxes = [len(a) for a in anchors]

    netin = Input((-1, -1, input_channels))

    net = netin
    grids = []
    prediction_layer_i = 0
    predictions = []
    layers = []
    for i, config in enumerate(architecture):
        if 'predict' in config['name']:
            with K.name_scope('predict{}'.format(prediction_layer_i)):
                inference = Conv2D(n_boxes[prediction_layer_i] * (n_polygon + 1), kernel_size=(1, 1),
                                   strides=(1, 1), name='predictor{}'.format(prediction_layer_i))(
                    net)
                prediction_layer_i += 1
                reshape = Reshape((-1, n_polygon + 1))(inference)
                prediction = Netout(n_polygon)(reshape)
                predictions.append(prediction)
                layers.append(inference)
                grid = K.int_shape(net)[-3], K.int_shape(net)[-2]
                grids.append(grid)
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

    meta_t = K.constant(GateNetEncoder.generate_anchors(grids, anchors, n_polygon),
                        K.tf.float32)

    netout = ConcatMeta(meta_t)(predictions)
    model = Model(netin, netout)
    return model