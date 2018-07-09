from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, TimeDistributed, SeparableConv2D, \
    DepthwiseConv2D


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


def time_dist_conv_leaky_creator(netin, config):
    return time_dist_conv_leaky(netin, config['filters'], config['kernel_size'], config['strides'], config['alpha'])


def time_dist_conv_leaky(netin, filters, kernel_size, strides, alpha):
    conv = TimeDistributed(Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False))(
        netin)
    norm = TimeDistributed(BatchNormalization())(conv)
    act = TimeDistributed(LeakyReLU(alpha=alpha))(norm)
    return act


def time_dist_max_pool_creator(netin, config):
    return time_dist_max_pool(netin, config['size'])


def time_dist_max_pool(netin, size):
    return TimeDistributed(MaxPooling2D(size))(netin)


def sep_conv_leaky_creator(netin, config):
    return conv_leaky(netin, config['filters'], config['kernel_size'], config['strides'], config['alpha'])


def sep_conv_leaky(netin, filters, kernel_size, strides, alpha):
    conv1 = DepthwiseConv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)
    conv2 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)
    return act2


layers = {'conv_leaky': conv_leaky_creator,
          'sep_conv_leaky': sep_conv_leaky_creator,
          'max_pool': max_pool_creator,
          'time_dist_conv_leaky': time_dist_conv_leaky_creator,
          'time_dist_max_pool': time_dist_max_pool_creator}
