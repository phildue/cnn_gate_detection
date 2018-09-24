from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.TransformDistort import TransformDistort
from utils.workdir import cd_work

image_source = ['resource/ext/samples/daylight_course1',
                'resource/ext/samples/daylight_course5',
                'resource/ext/samples/daylight_course3',
                'resource/ext/samples/iros2018_course1',
                'resource/ext/samples/iros2018_course5',
                'resource/ext/samples/iros2018_flights',
                'resource/ext/samples/real_and_sim',
                'resource/ext/samples/basement20k',
                'resource/ext/samples/basement_course3',
                'resource/ext/samples/basement_course1',
                'resource/ext/samples/iros2018_course3_test']
cd_work()
generator = GateGenerator(directories=image_source, batch_size=10, img_format='jpg',
                          shuffle=True, color_format='bgr', label_format='xml', start_idx=0)

batch = next(iter(generator.generate()))

create_dirs(['resource/ext/samples/all_distorted'])
set_writer = DatasetParser.get_parser('resource/ext/samples/all_distorted/', image_format='jpg', label_format='xml',
                                      start_idx=0, color_format='bgr')

dist_model = TransformDistort(BarrelDistortion.from_file('resource/demo_distortion_model.pkl'))

img_dist = []
label_dist = []
steps = int(generator.n_samples / 10)
gen = iter(generator.generate())
for i in range(steps):
    print('{}/{}'.format(i, steps))
    batch = next(gen)
    for img, label, image_file in batch:
        img_d, label_d = dist_model.transform(img, label)
        show(img_d, labels=label_d, t=1)
        img_dist.append(img_d)
        label_dist.append(label_d)

    set_writer.write(img_dist, label_dist)
