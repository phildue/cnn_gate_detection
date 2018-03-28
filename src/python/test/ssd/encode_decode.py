from modelzoo.models.ssd.SSD import SSD
from utils.BoundingBox import BoundingBox
from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show
from utils.workdir import cd_work

cd_work()
batch_size = 20
dataset = VocGenerator(batch_size=batch_size, shuffle=False, start_idx=2).generate()
batch = next(dataset)

ssd = SSD.ssd_test()
label_resized = []
for i in range(batch_size):
    img, label_res = resize(batch[i][0], (300, 300), label=batch[i][1])
    label_resized.append(label_res)

labels1_t = ssd.encoder.encode_label_batch(label_resized)

for i in range(batch_size):
    img = batch[i][0]
    label_true = batch[i][1]
    img, label_true = resize(img, (300, 300), label=label_true)
    label_dec = ssd.decoder.decode_netout_to_label(labels1_t[i])
    show(img, labels=[label_true], colors=[COLOR_GREEN], name='truth')
    show(img, labels=[label_dec], colors=[COLOR_RED], name='decoded')
    boxes = ssd.decoder.decode_netout_to_boxes(labels1_t[i])
    boxes = [b for b in boxes if b.c > 0.05]
    label_ = BoundingBox.to_label(boxes)
    show(img, labels=[label_], colors=[COLOR_RED], name='filtered')
