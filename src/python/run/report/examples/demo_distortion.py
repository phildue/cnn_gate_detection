from utils.imageprocessing.Backend import imread, imwrite
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
example_path = 'doc/thesis/fig/gate_example'
img = imread(example_path + '.jpg', 'bgr')

barrel = BarrelDistortion(img.shape[:2], rad_dist_params=[0.8, 0], tangential_dist_params=[0.8, 0], scale=0.5)
barrel_u = BarrelDistortion(img.shape[:2], rad_dist_params=[0.8, 0], tangential_dist_params=[0.8, 0], scale=1.0)
# barrel = BarrelDistortion.from_file('resource/demo_distortion_model.pkl')
print(barrel)
img_d = barrel.distort(img)
img_u = barrel_u.undistort(img_d)
show(img, name='Not Distorted')
show(img_d, name='Distorted')
show(img_u, name='Undistorted')

imwrite(img_d, example_path + '_distorted.jpg')
imwrite(img_u, example_path + '_undistorted.jpg')
barrel.save('resource/demo_distortion_model.pkl')