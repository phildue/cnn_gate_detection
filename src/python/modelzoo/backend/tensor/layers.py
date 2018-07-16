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


def dconv_creator(netin, config):
    return conv_leaky(netin, config['filters'], config['kernel_size'], config['strides'], config['alpha'])


def dconv(netin, filters, kernel_size, strides, alpha):
    conv1 = DepthwiseConv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)
    conv2 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)
    return act2


def bottleneck_dconv_creator(netin, config):
    return bottleneck_dconv(netin, config['filters'], config['kernel_size'], config['strides'], config['expansion'],
                            config['alpha'])


def bottleneck_dconv(netin, filters, kernel_size, strides, expansion, alpha):
    expand = Conv2D(int(K.int_shape(netin)[-1] * expansion), (1, 1), strides=(1, 1), padding='same', use_bias=False)(
        netin)
    norm1 = BatchNormalization()(expand)
    act1 = LeakyReLU(alpha=alpha)(norm1)

    dconv = DepthwiseConv2D(int(K.int_shape(netin)[-1] * expansion), kernel_size=kernel_size, strides=strides,
                            padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(dconv)
    act2 = LeakyReLU(alpha=alpha)(norm2)

    compress = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(
        act2)
    norm3 = BatchNormalization()(compress)

    return norm3


def bottleneck_dconv_residual_creator(netin, config):
    return bottleneck_dconv_residual(netin, config['filters'], config['kernel_size'], config['strides'],
                                     config['expansion'], config['alpha'])


def bottleneck_dconv_residual(netin, filters, kernel_size, strides, compression, alpha):
    fork = bottleneck_dconv(netin, filters, kernel_size, strides, compression, alpha)
    join = Add()([netin, fork])
    return join


def bottleneck_conv_creator(netin, config):
    return bottleneck_conv(netin, config['filters'], config['kernel_size'], config['strides'], config['compression'],
                           config['alpha'])


def bottleneck_conv(netin, filters, kernel_size, strides, compression, alpha):
    compress = Conv2D(int(K.int_shape(netin)[-1] * compression), (1, 1), strides=(1, 1), padding='same',
                      use_bias=False)(
        netin)
    norm1 = BatchNormalization()(compress)

    conv = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                  padding='same', use_bias=False)(norm1)
    norm2 = BatchNormalization()(conv)
    act = LeakyReLU(alpha=alpha)(norm2)

    return act


def bottleneck_conv_residual_creator(netin, config):
    return bottleneck_conv_residual(netin, config['filters'], config['kernel_size'], config['strides'],
                                    config['compression'], config['alpha'])


def bottleneck_conv_residual(netin, filters, kernel_size, strides, compression, alpha):
    fork = bottleneck_conv(netin, filters, kernel_size, strides, compression, alpha)
    join = Add()([netin, fork])
    return join


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
    return wr_bottleneck_conv_leaky(netin, config['filters'], config['compression'], config['kernel_size'],
                                    config['strides'],
                                    config['alpha'])


def wr_bottleneck_conv_leaky(netin, filters, compression_factor, kernel_size, strides, alpha):
    conv1 = Conv2D(int((K.int_shape(netin))[-1] * compression_factor), kernel_size=(1, 1), strides=strides,
                   padding='same', use_bias=False)(
        netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)

    conv2 = Conv2D(int(filters), kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)

    join = Add()([netin, act2])
    return join


def wr_inception_conv_leaky_creator(netin, config):
    return wr_inception_conv_leaky(netin, config['filters'], config['compression'], config['kernel_size'],
                                   config['strides'],
                                   config['alpha'])


def wr_inception_conv_leaky(netin, filters, compression, kernel_size, strides, alpha):
    conv_squeeze = Conv2D(int((K.int_shape(netin))[-1] * compression), kernel_size=(1, 1), strides=strides,
                          padding='same', use_bias=False)(netin)
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
    join = Add()([netin, concat])
    return join


layers = {'conv_leaky': conv_leaky_creator,
          'dconv': dconv_creator,
          'bottleneck_conv': bottleneck_conv_creator,
          'bottleneck_conv_residual': bottleneck_conv_residual_creator,
          'bottleneck_dconv': bottleneck_dconv_creator,
          'bottleneck_dconv_residual': bottleneck_dconv_residual_creator,
          'max_pool': max_pool_creator,
          'time_dist_conv_leaky': time_dist_conv_leaky_creator,
          'time_dist_max_pool': time_dist_max_pool_creator,
          'wr_basic_conv_leaky': wr_basic_conv_leaky_creator,
          'wr_bottleneck_conv_leaky': wr_bottleneck_conv_leaky_creator,
          'wr_inception_conv_leaky': wr_inception_conv_leaky_creator
          }
