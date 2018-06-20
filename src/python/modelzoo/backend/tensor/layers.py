from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D


def create_layer(netin, config):
    return layers[config['name']](netin, config)


def conv_leaky_creator(netin, config):
    return conv_leaky(netin, config['filters'], config['kernel_size'], config['strides'], config['alpha'])


def conv_leaky(netin, filters, kernel_size, strides, alpha):
    conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm = BatchNormalization()(conv)
    act = LeakyReLU(alpha=alpha)(norm)
    return act


def max_pool_creator(netin, config):
    return max_pool(netin, config['size'])


def max_pool(netin, size):
    return MaxPooling2D(size)(netin)


layers = {'conv_leaky': conv_leaky_creator,
          'max_pool': max_pool_creator}
