from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.utils import create_dirs
from utils.workdir import cd_work

cd_work()
src_dir = 'resource/ext/samples/'
target_dir = 'resource/ext/samples/yolo/flight_bgr/'

sets = [src_dir + 'daylight_course1',
        src_dir + 'daylight_course5',
        src_dir + 'daylight_course3',
        src_dir + 'iros2018_course1',
        src_dir + 'iros2018_course5',
        src_dir + 'basement_course3',
        src_dir + 'basement_course1',
        src_dir + 'iros2018_course3_test']

start_idx = 0
create_dirs([target_dir])
for dataset in sets:
    reader = DatasetParser.get_parser(directory=src_dir + dataset,
                                      label_format='xml',
                                      color_format='bgr',
                                      image_format='jpg'
                                      )
    writer = DatasetParser.get_parser(directory=target_dir,
                                      label_format='yolo',
                                      color_format='bgr',
                                      start_idx=start_idx)

    images, labels = reader.read(10)
    writer.write(images, labels)

    start_idx += len(images)
