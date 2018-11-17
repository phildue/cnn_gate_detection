from utils.imageprocessing.Backend import imread, imwrite
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.TransformMotionBlur import TransformMotionBlur
from utils.workdir import cd_work

cd_work()
example_path = 'doc/thesis/fig/gate_example'
img = imread(example_path + '.jpg', 'bgr')
sigma = 5.0

img_v, _ = TransformMotionBlur('vertical', sigma).transform(img)

img_h, _ = TransformMotionBlur('horizontal', sigma).transform(img)
show(img, name='Original')
show(img_v, 'vertical')
show(img_h, 'horizontal')

imwrite(img_v, example_path + '_motionblur_v.jpg')
imwrite(img_h, example_path + '_motionblur_h.jpg')
