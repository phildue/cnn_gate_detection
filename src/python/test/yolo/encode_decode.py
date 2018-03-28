from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show
from utils.workdir import cd_work

cd_work()
batch_size = 20
dataset = VocGenerator(batch_size=batch_size, shuffle=True).generate()
batch = next(dataset)

yolo = Yolo.tiny_yolo(norm=(208, 416), grid=(6, 13))
batch = [resize(b[0], (208, 416), label=b[1]) for b in batch]
labels1_enc = yolo.encoder.encode_label_batch([b[1] for b in batch])

for i in range(batch_size):
    img = batch[i][0]
    label_true = batch[i][1]
    img, label_true = resize(img, (208, 416), label=label_true)
    label_dec = yolo.postprocessor.decoder.decode_netout_to_label(labels1_enc[i])
    show(img, labels=[label_true], colors=[COLOR_GREEN], name='True')
    show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded')
