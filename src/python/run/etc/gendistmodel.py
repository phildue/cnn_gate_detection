from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.timing import tic
from utils.workdir import work_dir

work_dir()
dataset = GateGenerator(directories=['resource/ext/samples/cyberzoo/'],
                        batch_size=100, color_format='bgr',
                        shuffle=False, start_idx=0, valid_frac=0,
                        label_format='xml').generate()
batch = next(dataset)

camera_calibration = load_file('resource/cam_params_bebop.pkl')

distortion = camera_calibration.distortion[0]
# k1 = distortion[0]
# k2 = distortion[1]
# p1 = distortion[2]
# p2 = distortion[3]
k1 = 0.1
k2 = 0
p1 = 0
p2 = 0
print("Distortion", distortion)

distortion_model = BarrelDistortion(img_shape=(160, 315),
                                    rad_dist_params=[k1, k2],
                                    tangential_dist_params=[p1, p2],
                                    max_iterations=100,
                                    distortion_radius=1,
                                    conv_thresh=0.0000001,
                                    scale=0.95)
distortion_model.save('distortion_model_est.pkl')

distortion_model = BarrelDistortion.from_file('distortion_model_est.pkl')
for img, label, _ in batch:
    # img, label = resize(img, (80, 166), label=label)

    tic()
    img_d, label_d = distortion_model.undistort(img, label)

    img, label = resize(img, (320, 624), label=label)
    img_d, label_d = resize(img_d, (320, 624), label=label_d)
    show(img, labels=label, t=1)
    show(img_d, labels=label_d, name='undistorted')
