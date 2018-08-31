import random

from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.labelparser.YoloParser import YoloParser
from utils.fileaccess.utils import create_dirs
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc
from utils.workdir import cd_work


def filter(label):
    min_obj_size = 0.01
    max_obj_size = 1.0
    img_res = (416, 416)
    max_angle = 30
    objs_in_size = [obj for obj in label.objects if
                    min_obj_size < (obj.height * obj.width) / (img_res[0] * img_res[1]) < max_obj_size]

    max_aspect_ratio = 1.05 / (max_angle / 90)
    objs_within_angle = [obj for obj in objs_in_size if obj.height / obj.width < max_aspect_ratio]

    objs_in_view = []
    for obj in objs_within_angle:
        mat = obj.gate_corners.mat
        if (len(mat[(mat[:, 0] < 0) | (mat[:, 0] > img_res[1])]) +
            len(mat[(mat[:, 1] < 0) | (mat[:, 1] > img_res[0])])) > 2:
            continue
        objs_in_view.append(obj)

    return ImgLabel(objs_in_size)


cd_work()
set_name = 'yolo_flight_bgr'
out_dir = 'resource/ext/samples/' + set_name
src_dir = 'resource/ext/samples/'

sets = ['daylight_course1',
        'daylight_course5',
        'daylight_course3',
        'iros2018_course1',
        'iros2018_course5',
        'basement_course3',
        'basement_course1',
        'iros2018_course3_test']

create_dirs([out_dir])
batch_size = 8
generator = GateGenerator([src_dir + s for s in sets], batch_size,
                          color_format='bgr', shuffle=True,
                          label_format='xml',
                          img_format='jpg',
                          start_idx=0,
                          valid_frac=0,
                          filter=filter,
                          max_empty=0.0,
                          forever=False)
reader = generator.generate()
n_images = generator.n_samples

writer = YoloParser(out_dir, color_format='bgr', image_format='jpg', img_norm=(208, 208))
n_images_filtered = 0

for i in range(0, n_images, batch_size):
    tic()
    try:
        batch = next(reader)
        imgs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        writer.write(imgs, labels)
        toc(str(i) + '/' + str(n_images) + ' samples processed in ')
        n_images_filtered += batch_size
    except StopIteration:
        break

darknet_rel = '/home/phil/dronevision/'
indeces = [i for i in range(n_images_filtered)]
random.shuffle(indeces)
valid_set = indeces[:int(0.1 * len(indeces))]
train_set = indeces[int(0.1 * len(indeces)):]

train_set_path = out_dir + '/train.txt'
valid_set_path = out_dir + '/valid.txt'

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
