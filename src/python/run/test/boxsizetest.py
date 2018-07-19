from utils.fileaccess.GateGenerator import GateGenerator
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
for _ in range(iterations):
    batch = next(iterator)
    img_size = 640 * 480
    for img, label, _ in batch:
        for obj in label.objects:
            if obj.area > 0.25 * img_size:
                print(label)
                show(img, labels=[label])
