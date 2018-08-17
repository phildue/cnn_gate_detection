from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.labelparser.XmlParser import XmlParser
from utils.fileaccess.utils import create_dirs
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work


def filter(label):
    return label
    max_aspect_ratio = 1.05 / (45 / 90)
    objs_within_angle = [obj for obj in label.objects if obj.height / obj.width < max_aspect_ratio]

    objs_in_view = []
    for obj in objs_within_angle:
        mat = obj.gate_corners.mat
        if (len(mat[(mat[:, 0] < 0) | (mat[:, 0] > 416)]) +
            len(mat[(mat[:, 1] < 0) | (mat[:, 1] > 416)])) > 2:
            continue
        objs_in_view.append(obj)

    print("Objects: {}, bad angle: {}, out of view: {}, remaining: {}".format(len(label.objects),
                                                                              len(label.objects) - len(
                                                                                  objs_within_angle),
                                                                              len(label.objects) - len(objs_in_view),
                                                                              len(objs_in_view)))

    return ImgLabel(objs_in_view)


cd_work()
batch_size = 20
path = ['resource/ext/samples/daylight_course1']
outpath = 'resource/ext/samples/daylight_course1_filtered'

create_dirs([outpath])
gate_generator = GateGenerator(path, batch_size, color_format='bgr', shuffle=False, label_format='xml',
                               img_format='jpg',
                               filter=filter, remove_filtered=False, remove_empty=True)

writer = XmlParser(directory=outpath, color_format='bgr')

it = iter(gate_generator.generate())

for i in range(int(gate_generator.n_samples / batch_size)):
    print("Batch: ", i)
    batch = next(it)

    filtered_imgs = [b[0] for b in batch]
    filtered_labels = [b[1] for b in batch]
    writer.write(filtered_imgs, filtered_labels)

print("Remaining: ", GateGenerator([outpath], batch_size, color_format='bgr', shuffle=False, label_format='xml',
                                   img_format='jpg').n_samples)
