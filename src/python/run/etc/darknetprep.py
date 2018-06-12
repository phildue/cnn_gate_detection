import random

from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.labelparser.YoloParser import YoloParser
from utils.fileaccess.utils import save_file, create_dirs
from utils.imageprocessing.Imageprocessing import show
from utils.timing import tic, toc
from utils.workdir import cd_work

cd_work()
set_name = 'gates-daylight-yolo'
out_dir = 'resource/ext/samples/' + set_name
create_dirs([out_dir])
batch_size = 8
n_images = 10000
reader = GateGenerator(['resource/ext/samples/daylight/'], batch_size, color_format='bgr', shuffle=False,
                       label_format='xml',
                       img_format='jpg',
                       start_idx=0, valid_frac=0).generate()

writer = YoloParser(out_dir, color_format='bgr', image_format='jpg', img_norm=(416, 416))

for i in range(0, n_images, batch_size):
    tic()
    batch = next(reader)
    imgs = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    writer.write(imgs, labels)
    toc('Batch ' + str(i) + ' processed in ')

darknet_rel = '../../'
indeces = [i for i in range(n_images)]
random.shuffle(indeces)
valid_set = indeces[:int(0.1 * len(indeces))]
train_set = indeces[int(0.1 * len(indeces)):]

train_set_path = out_dir + '/gates-daylight.train.txt'
valid_set_path = out_dir + '/gates-daylight.valid.txt'

with open(valid_set_path, 'w') as f:
    for i in valid_set:
        f.write('{0:s}{1:s}/{2:05d}.jpg\n'.format(darknet_rel, out_dir, i))

with open(train_set_path, 'w') as f:
    for i in train_set:
        f.write('{0:s}{1:s}/{2:05d}.jpg\n'.format(darknet_rel, out_dir, i))

with open(out_dir + '/' + set_name + '.txt', 'w') as f:
    f.write('classes= 1\n')
    f.write('train= ' + darknet_rel + train_set_path + '\n')
    f.write('valid= ' + darknet_rel + valid_set_path + '\n')
    f.write('names= data/gate.names.list\n')
    f.write('backup= backup/\n')
