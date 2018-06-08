from modelzoo.models.cornernet.CornerNet import CornerNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize

dataset = GateGenerator(["resource/ext/samples/industrial_new/"], batch_size=batch_size,
                        color_format='bgr', label_format='xml', n_samples=batch_size).generate()
batch = next(dataset)

predictor = CornerNet((64, 64), 4)

batch = [resize(b[0], predictor.input_shape, label=b[1]) for b in batch]
labels1_enc = predictor.encoder.encode_label_batch([b[1] for b in batch])

for i in range(batch_size):
    img = batch[i][0]
    label_true = batch[i][1]
    img, label_true = resize(img, predictor.input_shape, label=label_true)
    label_dec = predictor.postprocessor.decoder.decode_netout_to_label(labels1_enc[i])
    show(img, labels=[label_true], colors=[COLOR_GREEN], name='True')
    show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded')
