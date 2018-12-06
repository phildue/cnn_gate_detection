from utils.fileaccess.labelparser.XmlParser import XmlParser
from utils.imageprocessing.Backend import imread, crop
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
example_path = 'resource/ext/samples/daylight_course1/00274'
img = imread(example_path + '.jpg', 'bgr')
label = XmlParser.read_label(example_path + '.xml')
# barrel = BarrelDistortion.from_file('resource/camera_calibration/distortion_model_est.pkl')
barrel = BarrelDistortion((416, 416), rad_dist_params=[0.7, 0], tangential_dist_params=[0.7, 0])
# barrel_u = BarrelDistortion(img.shape[:2], rad_dist_params=[0.7, 0], tangential_dist_params=[0.7, 0], scale=1.0)
# barrel = BarrelDistortion.from_file('resource/demo_distortion_model.pkl')
print(barrel)
img_d, label_d = barrel.distort(img, label, scale=1.0)
img_c, label_c = crop(img_d, (60, 60), (416-60, 416-60), label_d)
# img_d, label_d = barrel.distort(img, label, scale=1.4)
# img_c, label_c = resize(img_c, (240, 320), label=label_c)
# img_u, label_u = barrel.undistort(img_d, label_d, scale=1.0)
show(img, name='Not Distorted', labels=label)
show(img_c, name='crop', labels=label_c)
show(img_d, name='Distorted', labels=label_d)
# show(img_u, name='Undistorted', labels=label_d)

# imwrite(img_d, example_path + '_distorted.jpg')
# imwrite(img_u, example_path + '_undistorted.jpg')
# barrel.save('resource/demo_distortion_model.pkl')
