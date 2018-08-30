from utils.imageprocessing.Backend import imread, imwrite
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.TransformChromAbberr import TransformChromAbberr
from utils.imageprocessing.transform.TransformMotionBlur import TransformMotionBlur
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

cd_work()
example_path = 'doc/thesis/fig/gate_example'
img = imread(example_path + '.jpg', 'bgr')

img_t, _ = TransformChromAbberr(scale=(1, 1.01, 1), t_x=(-0.1, 0.1, 0.2), t_y=(0.1, -0.1, 0.01)).transform(img, )

show(img, name='Original')
show(img_t, 'Chromatic')

imwrite(img_t, example_path + '_chromatic.jpg')
