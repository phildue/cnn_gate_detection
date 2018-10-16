import numpy as np

from modelzoo.models.yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show, COLOR_BLUE
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

cd_work()
batch_size = 20

img_res = (416, 416)
yolo = Yolo.create_by_arch(norm=img_res,
                           anchors=np.array([[[81, 82],
                                              [135, 169],
                                              [344, 319]],
                                             [[10, 14],
                                              [23, 27],
                                              [37, 58]]]),
                           architecture=[
                               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'max_pool', 'size': (2, 2)},
                               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'max_pool', 'size': (2, 2)},
                               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'max_pool', 'size': (2, 2)},
                               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'max_pool', 'size': (2, 2)},
                               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'max_pool', 'size': (2, 2)},
                               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 1024, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 256, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'predict'},
                               {'name': 'route', 'index': [-4]},
                               {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 128, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'upsample', 'size': 2},
                               {'name': 'route', 'index': [-1, 8]},
                               {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1),
                                'alpha': 0.1},
                               {'name': 'predict'}], color_format='bgr', augmenter=None, class_names=[
        "muro"
    ])
dataset = GateGenerator(['resource/ext/samples/muro'], batch_size=batch_size, valid_frac=0,
                        color_format='bgr', label_format='xml', n_samples=100,
                        remove_filtered=False, max_empty=0.0).generate()
batch = next(dataset)
batch = [resize(b[0], (img_res[0], img_res[0]), label=b[1]) for b in batch]
labels1_enc = yolo.encoder.encode_label_batch([b[1] for b in batch])

for i in range(batch_size):
    img = batch[i][0]
    label_true = batch[i][1]
    img, label_true = resize(img, (img_res[0], img_res[1]), label=label_true)
    label_dec = yolo.postprocessor.decoder.decode_netout_to_label(labels1_enc[i])
    label_filtered = ImgLabel([b for b in label_dec.objects if b.confidence > 0])
    show(img, labels=[label_true], colors=[COLOR_GREEN], name='True')
    show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded')
    show(img, labels=[label_filtered], colors=[COLOR_BLUE], name='Filtered')
