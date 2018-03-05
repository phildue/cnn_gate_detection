import random

from frontend.models.ssd.SSD import SSD
from imageprocessing.Backend import resize
from imageprocessing.Imageprocessing import show
from labels.ObjectLabel import ObjectLabel
from workdir import work_dir

from src.python.modelzoo.models.ssd.utils import show_encoding
from src.python.utils.fileaccess import VocGenerator

work_dir()

dataset = VocGenerator(batch_size=100).generate()
class_names = ['B'] + ObjectLabel.classes.copy()

batch = next(dataset)
while True:
    idx = random.randint(0, 99)

    img_height, img_width = 300, 300
    ssd = SSD.ssd_test(conf_thresh=0, image_shape=(img_height, img_width, 3))
    label_true = batch[idx][1]
    img_org = batch[idx][0]
    img, label_true = resize(img_org, label=label_true, shape=(300, 300))
    label_true_t = ssd.encoder.encode_label(label_true)
    ObjectLabel.classes = class_names
    label_true_t[:, -4:] = ssd.decoder.decode_coord(label_true_t[:, -4:])
    show_encoding(label_true_t, img)
    show(img, labels=[label_true], thickness=3, name='Truth')
