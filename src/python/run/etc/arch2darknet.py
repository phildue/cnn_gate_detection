import numpy as np

from utils.fileaccess.utils import save_file, create_dirs
from utils.workdir import cd_work

cd_work()
name = 'd0'
img_res = (416, 416)
arch = [
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 4, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2), 'strides': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 8, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2), 'strides': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2), 'strides': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 24, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2), 'strides': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2), 'strides': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'predict'},
    {'name': 'route', 'index': [-2]},
    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'upsample', 'size': 2, 'strides': (2, 2)},
    {'name': 'route', 'index': [-1, 8]},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'predict'}
]

anchors = np.array([
    [[81, 82],
     [135, 169],
     [344, 319]],
    [[10, 14],
     [23, 27],
     [37, 58]],
])
n_anchors = 6

mask = ['3,4,5',
        '0,1,2']

anchors_str = ''
for batch in anchors:
    for row in batch:
        anchors_str += ' '
        for el in row:
            anchors_str += str(el)
            anchors_str += ','
anchors_str = anchors_str[:-1]

darknet_network = []
i_predictor = 0
for i, l in enumerate(arch):
    if l['name'] == 'conv_leaky':
        darknet_network.append(
            '[convolutional]\n'
            'batch_normalize=1\n'
            'filters={}\n'
            'size={}\n'
            'stride={}\n'
            'pad=1\n'
            'activation=leaky'.format(
                l['filters'],
                l['kernel_size'][0],
                l['strides'][0]
            )
        )
    elif l['name'] == 'max_pool':
        darknet_network.append(
            '[maxpool]\n'
            'size={}\n'
            'stride={}\n'.format(
                l['size'][0],
                l['strides'][0]
            )
        )
    elif l['name'] == 'predict':

        darknet_network.append(
            '[convolutional]\n'
            'size=1\n'
            'stride=1\n'
            'pad=1\n'
            'filters={}\n'
            'activation=linear'.format(
                len(anchors[i_predictor]) * 6
            ))
        darknet_network.append(
            '[yolo]\n' \
            'mask ={}\n' \
            'anchors ={}\n' \
            'classes=1\n' \
            'num={}\n' \
            'jitter=.3\n' \
            'ignore_thresh=.7\n' \
            'truth_thresh=1\n' \
            'random=1'.format(
                mask[i_predictor],
                anchors_str,
                n_anchors
            )
        )
        i_predictor += 1
    elif l['name'] == 'route':
        layers_str = ''
        for i_idx, idx in enumerate(l['index']):
            layers_str += str(idx)
            layers_str += ','
        layers_str = layers_str[:-1]

        darknet_network.append(
            '[route]\n'
            'layers={}'.format(
                layers_str
            )
        )
    elif l['name'] == 'upsample':
        darknet_network.append(
            '[upsample]\n'
            'stride={}'.format(
                l['strides'][0]
            )
        )

darknet_cfg = '[net]\n' \
              '# Testing\n' \
              'batch=1\n' \
              'subdivisions=1\n' \
              '# Training\n' \
              '# batch=64\n' \
              '# subdivisions=2\n' \
              'width={}\n' \
              'height={}\n' \
              'channels=3\n' \
              'momentum=0.9\n' \
              'decay=0.0005\n' \
              'angle=0\n' \
              'saturation = 1.5\n' \
              'exposure = 1.5\n' \
              'hue=.1\n' \
              '\n' \
              'learning_rate=0.001\n' \
              'burn_in=1000\n' \
              'max_batches = 500200\n' \
              'policy=steps\n' \
              'steps=400000,450000\n' \
              'scales=.1,.1\n\n'.format(
    img_res[0],
    img_res[1],
)

for l in darknet_network:
    darknet_cfg += l
    darknet_cfg += '\n\n'

print(darknet_cfg)
create_dirs(['out/darknet_cfg/'])
save_file(darknet_cfg, name + '.cfg', '../dronerace2018/target/jevois/share/darknet/yolo/cfg/')
