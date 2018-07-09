from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, TimeDistributed, Add, Concatenate
import keras.backend as K
from modelzoo.backend.tensor.DepthwiseConv2D import DepthwiseConv2D


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


def wr_basic_conv_leaky_creator(netin, config):
    return wr_basic_conv_leaky(netin, config['filters'], config['kernel_size'],
                               config['strides'], config['alpha'])


def wr_basic_conv_leaky(netin, filters, kernel_size, strides, alpha):
    conv1 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)

    conv2 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)

    return Add()([netin, act2])


def wr_bottleneck_conv_leaky_creator(netin, config):
    return wr_bottleneck_conv_leaky(netin, config['filters'], config['compression'], config['kernel_size'], config['strides'],
                                    config['alpha'])


def wr_bottleneck_conv_leaky(netin, filters, compression_factor, kernel_size, strides, alpha):
    conv1 = Conv2D(int(filters * compression_factor), kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)(
        netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)

    conv2 = Conv2D(int(filters), kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)

    conv3 = Conv2D((K.int_shape(netin))[-1], kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)(act2)
    norm3 = BatchNormalization()(conv3)
    act3 = LeakyReLU(alpha=alpha)(norm3)

    return Add()([netin, act3])


def wr_inception_conv_leaky_creator(netin, config):
    return wr_inception_conv_leaky(netin, config['filters'], config['compression'], config['kernel_size'], config['strides'],
                                   config['alpha'])


def wr_inception_conv_leaky(netin, filters,compression, kernel_size, strides, alpha):
    conv_squeeze = Conv2D(int(filters * compression), kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)(netin)
    norm_squeeze = BatchNormalization()(conv_squeeze)
    act_squeeze = LeakyReLU(alpha=alpha)(norm_squeeze)

    conv11 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        act_squeeze)
    norm11 = BatchNormalization()(conv11)
    act11 = LeakyReLU(alpha=alpha)(norm11)

    conv12 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(act11)
    norm12 = BatchNormalization()(conv12)
    act12 = LeakyReLU(alpha=alpha)(norm12)

    conv21 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        act_squeeze)
    norm21 = BatchNormalization()(conv21)
    act21 = LeakyReLU(alpha=alpha)(norm21)

    concat = Concatenate()([act12, act21])

    conv_expand = Conv2D(K.int_shape(netin)[-1], kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)(concat)
    norm_expand = BatchNormalization()(conv_expand)
    act_expand = LeakyReLU(alpha=alpha)(norm_expand)

    return Add()([netin, act_expand])


layers = {'conv_leaky': conv_leaky_creator,
          'sep_conv_leaky': sep_conv_leaky_creator,
          'max_pool': max_pool_creator,
          'time_dist_conv_leaky': time_dist_conv_leaky_creator,
          'time_dist_max_pool': time_dist_max_pool_creator,
          'wr_basic_conv_leaky': wr_basic_conv_leaky_creator,
          'wr_bottleneck_conv_leaky': wr_bottleneck_conv_leaky_creator,
          'wr_inception_conv_leaky': wr_inception_conv_leaky_creator
          }
