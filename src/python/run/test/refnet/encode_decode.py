from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.refnet.RefNet import RefNet
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work
import numpy as np

cd_work()
batch_size = 1
anchors = np.array([[[1, 1],
                     [0.3, 0.3],
                     ]])
predictor = RefNet.create_by_arch(
    architecture=[{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                  {'name': 'max_pool', 'size': (2, 2)},
                  {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1}],
    anchors=anchors,
    n_rois=2)

dataset = GateGenerator(["resource/ext/samples/industrial_new/"], batch_size=batch_size,
                        color_format='bgr', label_format='xml', n_samples=99).generate()
batch = next(dataset)

batch = [resize(b[0], predictor.input_shape, label=b[1]) for b in batch]
labels1_enc = predictor.encoder.encode_label_batch([b[1] for b in batch], [b[0] for b in batch])
for i in range(batch_size):
    img = batch[i][0]
    label_true = batch[i][1]
    print("Objects: {}".format(len(label_true.objects)))
    print(np.round(labels1_enc[i], 2))
    print('_____________________________')
    img, label_true = resize(img, predictor.input_shape, label=label_true)
    label_dec = predictor.postprocessor.decoder.decode_netout_to_label(labels1_enc[i])
    label_filt = ImgLabel([obj for obj in label_dec.objects if obj.confidence > 0])
    print(label_filt)
    show(img, labels=[label_true], colors=[COLOR_GREEN], name='True', t=1)
    show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded', t=1)
    show(img, labels=[label_filt], colors=[COLOR_RED], name='Filtered')
