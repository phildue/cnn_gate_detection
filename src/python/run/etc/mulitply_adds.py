import numpy as np


def o_conv(volume, layer):
    sy, sx = layer['strides']
    filters = layer['filters']
    kernel_h, kernel_w = layer['kernel_size']
    input_h, input_w, input_ch = volume
    pad_y, pad_x = kernel_h - 1, kernel_w - 1

    output_volume = (input_h - (kernel_h - 1) + pad_y) / sy, (input_w - (kernel_w - 1) + pad_x) / sx, filters

    multiply_adds = output_volume[0] * output_volume[1] * input_ch * kernel_h * kernel_w

    return output_volume, multiply_adds


def o_pool(volume, layer):
    size = layer['size']

    output_volume = np.ceil(volume[0] / size[0]), np.ceil(volume[1] / size[1]), volume[2]

    multiply_adds = output_volume[0] * output_volume[1] * 1

    return output_volume, multiply_adds


def main(architecture, volume):
    lookup = {
        'conv_leaky': o_conv,
        'max_pool': o_pool,
    }
    multiply_adds_total = 0

    for layer in architecture:
        print(volume)
        volume, multiply_adds = lookup[layer['name']](volume, layer)
        multiply_adds_total += multiply_adds
        print("Multiply-Adds Layer: {}".format(multiply_adds))
        print("Multiply-Adds Total: {}".format(multiply_adds_total))


if __name__ == '__main__':
    architecture = [
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
    volume = (416, 416, 3)

    main(architecture, volume)
