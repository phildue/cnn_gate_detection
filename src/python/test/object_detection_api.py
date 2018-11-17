from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.utils import create_dirs
from utils.workdir import cd_work

cd_work()
out_dir = 'resource/ext/samples/industrial_room_test_tf'
create_dirs([out_dir])
generator = GateGenerator(directories=['resource/ext/samples/industrial_room_test'],
                          batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=0.1,
                          label_format='xml',
                          )

it = iter(generator.generate())
tf_writer = DatasetParser.get_parser(directory=out_dir,
                                     label_format='tf_record',color_format='bgr')
for i in range(1):
    batch = next(it)
    imgs = []
    labels = []
    for img, label, _ in batch:
        imgs.append(img)
        labels.append(label)
    tf_writer.write(imgs,labels)