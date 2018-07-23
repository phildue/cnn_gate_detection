import numpy as np


def conv(volume, layer):
    sy, sx = layer['strides']
    filters = layer['filters']
    kernel_h, kernel_w = layer['kernel_size']
    input_h, input_w, input_ch = volume
    pad_y, pad_x = kernel_h - 1, kernel_w - 1

    output_volume = (input_h - (kernel_h - 1) + pad_y) / sy, (input_w - (kernel_w - 1) + pad_x) / sx, filters

    multiply_adds = output_volume[0] * output_volume[1] * output_volume[2] * input_ch * kernel_h * kernel_w

    return output_volume, multiply_adds


def dconv(volume, layer):
    sy, sx = layer['strides']
    filters = layer['filters']
    kernel_h, kernel_w = layer['kernel_size']
    input_h, input_w, input_ch = volume
    pad_y, pad_x = kernel_h - 1, kernel_w - 1

    output_volume = (input_h - (kernel_h - 1) + pad_y) / sy, (input_w - (kernel_w - 1) + pad_x) / sx, filters

    multiply_adds = output_volume[0] * output_volume[1] * output_volume[2] * (input_ch + kernel_h * kernel_w)

    return output_volume, multiply_adds


def bottleneck_conv(volume, layer):
    sy, sx = layer['strides']
    filters = layer['filters']
    kernel_h, kernel_w = layer['kernel_size']
    input_h, input_w, input_ch = volume
    pad_y, pad_x = kernel_h - 1, kernel_w - 1
    compression = layer['compression']

    output_volume = (input_h - (kernel_h - 1) + pad_y) / sy, (input_w - (kernel_w - 1) + pad_x) / sx, filters

    multiply_adds = input_h * input_w * input_ch * input_ch * compression + output_volume[0] * output_volume[1] * (
            input_ch * compression * kernel_h * kernel_w)

    return output_volume, multiply_adds


def bottleneck_dconv(volume, layer):
    sy, sx = layer['strides']
    filters = layer['filters']
    kernel_h, kernel_w = layer['kernel_size']
    input_h, input_w, input_ch = volume
    pad_y, pad_x = kernel_h - 1, kernel_w - 1
    expand = layer['expand']

    output_volume = (input_h - (kernel_h - 1) + pad_y) / sy, (input_w - (kernel_w - 1) + pad_x) / sx, filters

    ops_expand = input_h * input_w * input_ch * input_ch * expand
    ops_dconv = input_h * input_w * input_ch * expand * kernel_w * kernel_h
    ops_compress = input_h * input_w * input_ch * expand * filters

    multiply_adds = ops_expand + ops_dconv + ops_compress

    return output_volume, multiply_adds


def bottleneck_dconv_residual(volume, layer):
    fork_volume, fork_ops = bottleneck_conv(volume, layer)

    ops = fork_ops + fork_volume[0] * fork_volume[1] * fork_volume[2]

    return fork_volume, ops


def bottleneck_conv_residual(volume, layer):
    fork_volume, fork_ops = bottleneck_dconv(volume, layer)

    ops = fork_ops + fork_volume[0] * fork_volume[1] * fork_volume[2]

    return fork_volume, ops


def max_pool(volume, layer):
    size = layer['size']

    output_volume = np.ceil(volume[0] / size[0]), np.ceil(volume[1] / size[1]), volume[2]

    multiply_adds = output_volume[0] * output_volume[1] * 1

    return output_volume, multiply_adds


def count_operations(architecture, volume_in, verbose=False):
    lookup = {
        'conv_leaky': conv,
        'max_pool': max_pool,
        'dconv': dconv,
        'bottleneck_conv': bottleneck_conv,
        'bottleneck_dconv': bottleneck_dconv,
        'bottleneck_conv_residual': bottleneck_conv_residual,
        'bottleneck_dconv_residual': bottleneck_dconv_residual
    }
    multiply_adds_total = 0

    collect = []
    for i, layer in enumerate(architecture):
        volume_out, multiply_adds = lookup[layer['name']](volume_in, layer)

        collect.append({'name': layer['name'], 'in': volume_in, 'out': volume_out, 'operations': multiply_adds})

        multiply_adds_total += multiply_adds
        if verbose:
            print("{}: {} {} --> {}".format(i, layer['name'], volume_out, volume_in))
            print("Ops Layer: {}".format(multiply_adds))
            print("Ops Total: {}".format(multiply_adds_total))

        volume_in = volume_out

    return collect

if __name__ == '__main__':
    network = [
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    ]
    img = (416, 416, 3)

    count_operations(network, img, True)
