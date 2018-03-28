import numpy as np

from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
image_source = ['resource/ext/samples/bebop20k']
batch_size = 8
height, width = 160, 315
data_generator = GateGenerator(image_source, batch_size=batch_size, valid_frac=0.1,
                               color_format='bgr', label_format=None)

X = np.zeros((data_generator.n_samples, height, width, 3), dtype=np.uint8)
it = iter(data_generator.generate())
for i in range(int(data_generator.n_samples / batch_size)):
    for img, label, _ in next(it):
        # show(img)
        X[i] = img.array

for i in range(X.shape[0]):
    pixel_min = np.min(np.min(X[i], 0), 0)
    pixel_max = np.max(np.max(X[i], 0), 0)
    contrast_stretched = (X[i] - pixel_min) / (pixel_max - pixel_min) * 255
    show(Image(contrast_stretched.astype(np.uint8), 'bgr'))
