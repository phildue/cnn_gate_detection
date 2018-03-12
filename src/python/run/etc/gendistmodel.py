from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.timing import tic, toc
from utils.workdir import work_dir

work_dir()
dataset = GateGenerator(directories=['resource/samples/bebop/'],
                        batch_size=100, color_format='bgr',
                        shuffle=False, start_idx=0, valid_frac=0,
                        label_format='xml').generate()
batch = next(dataset)

distortion_model = BarrelDistortion(img_shape=(80, 166),
                                    rad_dist_params=[0.1, 0],
                                    max_iterations=100,
                                    distortion_radius=1)
distortion_model.save('some_distortion.pkl')

for img, label, _ in batch:
    img, label = resize(img, (80, 166), label=label)

    tic()
    img_d, label_d = distortion_model.distort(img, label)

    img, label = resize(img, (320, 624), label=label)
    img_d, label_d = resize(img_d, (320, 624), label=label_d)
    show(img, labels=label, t=1)
    show(img_d, labels=label_d, name='distorted')
