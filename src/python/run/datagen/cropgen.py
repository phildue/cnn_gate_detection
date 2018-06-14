from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.utils import create_dirs
from utils.timing import tic, toc, tuc
from utils.workdir import cd_work

cd_work()
batch_size = 16
n_samples = 9000
out_path = 'resource/ext/samples/crop20k'
create_dirs([out_path])
image_source = ['resource/ext/samples/daylight', 'resource/ext/samples/industrial_new']
train_gen = CropGenerator(GateGenerator(image_source, batch_size=batch_size, valid_frac=0.0,
                                        color_format='bgr', label_format='xml'))

gen = iter(train_gen.generate())
set_writer = DatasetParser.get_parser(out_path, image_format='jpg', label_format='xml', start_idx=0,
                                      color_format='bgr')
tic()

for i in range(0, n_samples, batch_size):
    batch = next(gen)
    imgs = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    set_writer.write(imgs, labels)
    del imgs
    del labels
    tuc('{}/{} samples cropped '.format(i, n_samples))
