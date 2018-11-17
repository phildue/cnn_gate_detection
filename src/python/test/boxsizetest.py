from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.Backend import imwrite
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
batch_size = 50
generator = GateGenerator(directories=['resource/ext/samples/daylight_test/'],
                          batch_size=batch_size, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=0,
                          label_format='xml', img_format='jpg')
iterations = int(generator.n_samples / batch_size)
print(iterations)
iterator = generator.generate()
create_dirs(['out/examples/'])
j = 1
for i in range(iterations):
    batch = next(iterator)
    img_size = 640 * 480
    for img, label, _ in batch:
        for obj in label.objects:
            min_area = 0.001
            max_area = 0.05
            if min_area * img_size < obj.area < max_area * img_size:
                print(label)
                show(img, labels=[label])
                imwrite(img, 'doc/report/2018-07-25/size_examples/{}-{}({}).jpg'.format(min_area, max_area, j))
                j += 1
