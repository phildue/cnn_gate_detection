import numpy as np

from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.Backend import imread, resize
from utils.workdir import cd_work

cd_work()
image_source = ['resource/samples/cyberzoo/']
output_dir = 'resource/samples/cyberzoo_cats'
create_dirs([output_dir])
batch_size = 50
img_shape = (180, 315)
data_generator = GateGenerator(image_source, batch_size=batch_size, valid_frac=0.0,
                               color_format='bgr', label_format='xml', img_format='jpg')
cat_image = imread('resource/cats.jpg', 'bgr')
cat_image = resize(cat_image, img_shape)

center_x = int(img_shape[1] / 2)
center_y = int(img_shape[0] / 2)

set_writer = SetFileParser(output_dir, 'jpg', 'xml')

it = iter(data_generator.generate())

iterations = int(data_generator.n_samples / batch_size)

for i in range(iterations):
    img_cats = []
    labels = []
    for img, label, img_path in next(it):
        for o in label.objects:
            y_min = o.y_min
            y_max = o.y_max
            x_min = o.x_min
            x_max = o.x_max
            w = x_max - x_min
            h = y_max - y_min
            y_start = np.max([int(center_y - h / 2), 0])
            y_end = np.min([int(center_y + h / 2), h])
            x_start = np.max([int(center_x - w / 2), 0])
            x_end = np.min([int(center_x + w / 2), w])
            patch = cat_image.array[y_start:y_end, x_start:x_end]

            img.array[y_min:y_max, x_min: x_max] = patch
            img_cats.append(img)
            labels.append(label)
    set_writer.write(img_cats, labels)
