import numpy as np

from modelzoo.models.cropnet import CropNet
from modelzoo.models.cropnet import CropNetBase
from modelzoo.models.cropnet.CropGridLoss import CropGridLoss
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show
from utils.labels.ImgLabel import ImgLabel
from utils.labels.utils import resize_label
from utils.workdir import cd_work

cd_work()
batch_size = 10

predictor = CropNet(net=CropNetBase(architecture=None, input_shape=(52, 52), loss=CropGridLoss()), input_shape=(52, 52),
                    output_shape=[(13, 13)], augmenter=None,
                    encoding='anchor')

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
    img, label_true = resize(img, (104, 104), label=label_true)
    label_dec = predictor.postprocessor.decoder.decode_netout_to_label(labels1_enc[i])
    label_pp = ImgLabel(objects=[o for o in label_dec.objects if o.confidence > 0])
    label_dec = resize_label(label_dec, predictor.input_shape, (104, 104))
    label_pp = resize_label(label_pp, predictor.input_shape, (104, 104))
    print(label_pp)
    show(img, labels=[label_true], colors=[COLOR_GREEN], name='True', t=1)
    show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded', t=1)
    show(img, labels=[label_pp], colors=[COLOR_RED], name='Postprocessed', t=0)
