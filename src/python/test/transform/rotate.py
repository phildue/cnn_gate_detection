from utils.imageprocessing.transform.RandomRotate import RandomRotate

from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
generator = GateGenerator(directories=['resource/samples/cyberzoo/'],
                          batch_size=100, color_format='bgr',
                          shuffle=True, start_idx=0, valid_frac=0,
                          label_format='xml', img_format='jpg')
batch = next(generator.generate())
for img, label, _ in batch:
    show(img, 'org', labels=label)

    img, label = RandomRotate(-10, 10).transform(img, label)
    show(img, 'rotation', labels=label)
