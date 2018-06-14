from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show
from utils.workdir import cd_work
import numpy as np

cd_work()
batch_size = 10

predictor = GateNet.create('GateNet3x3', batch_size=batch_size, norm=(52, 52), grid=[(3, 3)],
                           anchors=np.array([[1.28, 1.84],
                                             [3.39, 5.10],
                                             [2.29, 3.21],
                                             [7.60, 8.60],
                                             [4.25, 8.16]]))

dataset = CropGenerator(GateGenerator(["resource/ext/samples/industrial_new/"], batch_size=batch_size,
                                      color_format='bgr', label_format='xml', n_samples=99)).generate()
batch = next(dataset)

batch = [resize(b[0], predictor.input_shape, label=b[1]) for b in batch]
labels1_enc = predictor.encoder.encode_label_batch([b[1] for b in batch])
for i in range(batch_size):
    img = batch[i][0]
    label_true = batch[i][1]
    print("Objects: {}".format(len(label_true.objects)))
    print(np.round(labels1_enc[i], 2))
    print('_____________________________')
    # img, label_true = resize(img, predictor.input_shape, label=label_true)
    # label_dec = predictor.postprocessor.decoder.decode_netout_to_label(labels1_enc[i])
    # show(img, labels=[label_true], colors=[COLOR_GREEN], name='True')
    # show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded')
