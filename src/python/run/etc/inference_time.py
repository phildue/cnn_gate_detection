from run.etc.mulitply_adds import count_operations


def time_per_operation(img, layer, time):
    lookup = count_operations(layer, img)

    operations = sum([entry['operations'] for entry in lookup])

    t_op = time / operations

    return t_op


"""Narrow Stride"""
narrowStride = [
    {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 16, 'strides': (2, 2), 'alpha': 0.1},
]
imgs = (208, 208, 3), (104, 104, 3), (52, 52, 3)
times = 75.53, 19.86, 4.74
print("Conv2x2x16Conv2x2x16:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, narrowStride, time))

narrowStrideBottleneck = [
    {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 8, 'strides': (2, 2), 'alpha': 0.1},
]
imgs = (208, 208, 3), (104, 104, 3), (52, 52, 3)
times = 68.76, 18.58, 4.36
print("Conv2x2x16Conv2x2x8|2,2:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, narrowStrideBottleneck, time))

convMaxPool = [
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)}
]
imgs = (208, 208, 3), (104, 104, 3), (52, 52, 3)
times = 70.13, 17.9, 4.22
print("Conv3x3x16MaxPool:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, convMaxPool, time))

convStride = [
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (2, 2), 'alpha': 0.1},
]
imgs = (208, 208, 3), (104, 104, 3), (52, 52, 3)
times = 15.6, 3.8, 1.14
print("Conv3x3x16|2,2:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, convStride, time))

narrowStride = [
    {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 32, 'strides': (2, 2), 'alpha': 0.1},
]
imgs = (104, 104, 32), (52, 52, 32), (26, 26, 32)
times = 52.79, 14.2, 3.82
print("Conv2x2x32Conv2x2x32:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, narrowStride, time))

narrowStrideBottleneck = [
    {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 16, 'strides': (2, 2), 'alpha': 0.1},
]
imgs = (104, 104, 32), (52, 52, 32), (26, 26, 32)
times = 52.79, 14.2, 3.82
print("Conv2x2x32Conv2x2x16|2,2:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, narrowStrideBottleneck, time))

convMaxPool = [
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)}
]
imgs = (104, 104, 32), (52, 52, 32), (26, 26, 32)
times = 65.24, 18.9, 5.36
print("Conv3x3x32MaxPool:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, convMaxPool, time))

convMaxPool = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)}
]
imgs = ((208, 208, 3), (104, 104, 3), (52, 52, 3))
times = 144.1, 37, 9.7

print("Conv6x6x16MaxPool:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, convMaxPool, time))

convStride = [
    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (2, 2), 'alpha': 0.1},
]
imgs = ((208, 208, 3), (104, 104, 3), (52, 52, 3))
times = 35.6, 9.6, 2.6

print("Conv6x6x16MaxPool:")

for img, time in zip(imgs, times):
    print(time_per_operation(img, convStride, time))
