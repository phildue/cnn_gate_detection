from utils.fileaccess.labelparser.XmlParser import XmlParser
from utils.imageprocessing.Backend import imread, imwrite
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
example_path = 'resource/ext/samples/daylight_course1/00045'
img = imread(example_path + '.jpg', 'bgr')
label = XmlParser.read_label(example_path + '.xml')
barrel = BarrelDistortion((416, 416), rad_dist_params=[0.7, 0], tangential_dist_params=[0.7, 0])
# barrel_u = BarrelDistortion(img.shape[:2], rad_dist_params=[0.7, 0], tangential_dist_params=[0.7, 0], scale=1.0)
# barrel = BarrelDistortion.from_file('resource/demo_distortion_model.pkl')
print(barrel)
img_d = barrel.distort(img, scale=0.5)
img_u = barrel.undistort(img_d, scale=1.0)
show(img, name='Not Distorted')
show(img_d, name='Distorted')
show(img_u, name='Undistorted')

imwrite(img_d, example_path + '_distorted.jpg')
imwrite(img_u, example_path + '_undistorted.jpg')
# barrel.save('resource/demo_distortion_model.pkl')
