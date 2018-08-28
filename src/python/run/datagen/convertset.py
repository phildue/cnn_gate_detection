from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.utils import create_dirs
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc
from utils.workdir import cd_work

min_obj_size = 0.01
max_obj_size = 1.0
img_res = (416, 416)
max_angle = 30


def filter(label):
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
src_dir = 'resource/ext/samples/'
target_dir = 'resource/ext/samples/yolo/flight_bgr/'

sets = ['daylight_course1',
        'daylight_course5',
        'daylight_course3',
        'iros2018_course1',
        'iros2018_course5',
        'basement_course3',
        'basement_course1',
        'iros2018_course3_test']

start_idx = 0
create_dirs([target_dir])

max_empty_frac = 0.05

for dataset in sets:
    tic()
    reader = DatasetParser.get_parser(directory=src_dir + dataset,
                                      label_format='xml',
                                      color_format='bgr',
                                      image_format='jpg'
                                      )
    writer = DatasetParser.get_parser(directory=target_dir,
                                      label_format='yolo',
                                      color_format='bgr',
                                      start_idx=start_idx,
                                      )
    images, labels = reader.read()
    n_samples = len(images)
    max_empty = int(max_empty_frac * n_samples)

    images_filtered = []
    labels_filtered = []
    n_empty = 0
    for i in range(n_samples):
        label = labels[i]
        image = images[i]
        label_filtered = filter(label)
        if (len(label_filtered.objects) == 0 and n_empty < max_empty) or len(label_filtered.objects) > 0:
            images_filtered.append(image)
            labels_filtered.append(label_filtered)
            n_empty += 1

    writer.write(images_filtered, labels_filtered)
    toc(str(len(images)) + " read. " + str(len(images_filtered)) + " created.")

    start_idx += len(images)
