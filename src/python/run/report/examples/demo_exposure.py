from utils.imageprocessing.Backend import imread, imwrite
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.TransformChromAbberr import TransformChromAbberr
from utils.imageprocessing.transform.TransformExposure import TransformExposure
from utils.imageprocessing.transform.TransformMotionBlur import TransformMotionBlur
from utils.imageprocessing.transform.TransformOutOfFocusBlur import TransformOutOfFocusBlur
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

cd_work()
example_path = 'doc/thesis/fig/gate_example'
img = imread(example_path + '.jpg', 'bgr')

img_t, _ = TransformExposure(contrast=1.0, delta_exposure=2.0).transform(img)

show(img, name='Original')
show(img_t, 'Exposure')

imwrite(img_t, example_path + '_exposure.jpg')
