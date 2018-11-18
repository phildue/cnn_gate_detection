from utils.imageprocessing.Backend import imread, imwrite
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.TransformOutOfFocusBlur import TransformOutOfFocusBlur
from utils.workdir import cd_work

cd_work()
example_path = 'doc/thesis/fig/gate_example'
img = imread(example_path + '.jpg', 'bgr')

img_t, _ = TransformOutOfFocusBlur(kernel_size=(15, 15), sigmaX=2.0, sigmaY=2.0).transform(img)

show(img, name='Original')
show(img_t, 'Blur')

imwrite(img_t, example_path + '_focusblur.jpg')
