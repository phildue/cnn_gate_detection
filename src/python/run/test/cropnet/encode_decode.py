import numpy as np

from modelzoo.backend.tensor.CropLoss import CropLoss
from modelzoo.backend.tensor.cropnet.CropNet2L import CropNet2L
from modelzoo.models.cropnet.CropNet import CropNet
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show
from utils.workdir import cd_work

cd_work()
batch_size = 10

predictor = CropNet(net=CropNet2L(input_shape=(52, 52), loss=CropLoss()), augmenter=None)

dataset = GateGenerator(["resource/ext/samples/industrial_new/"], batch_size=batch_size,
                        color_format='bgr', label_format='xml', n_samples=99).generate()
batch = next(dataset)

batch = [resize(b[0], predictor.input_shape, label=b[1]) for b in batch]
labels1_enc = predictor.encoder.encode_label_batch([b[1] for b in batch])
for i in range(batch_size):
    img = batch[i][0]
    label_true = batch[i][1]
    print("Objects: {}".format(len(label_true.objects)))
    print(np.round(labels1_enc[i], 2))
    print('_____________________________')
    img, label_true = resize(img, predictor.input_shape, label=label_true)
    label_dec = predictor.postprocessor.decoder.decode_netout_to_label(labels1_enc[i])
    label_img = Image(label_dec * 255, 'bgr')
    label_img = resize(label_img, (52, 52))
    print(label_dec)
    show(img, labels=[label_true], colors=[COLOR_GREEN], name='True', t=1)
    show(label_img, name='Decoded')
