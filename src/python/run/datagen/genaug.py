import numpy as np
from modelzoo.augmentation.AugmenterEnsemble import AugmenterEnsemble

from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.transform.TransformDistort import TransformDistort
from utils.timing import tic, toc
from utils.workdir import cd_work

cd_work()
image_source = 'resource/samples/mult_gate_aligned_blur'
batch_size = 20
n_samples = 10000
sample_path = 'resource/samples/mult_gate_aligned_blur_distort/'
create_dirs([sample_path])
data_generator = GateGenerator(image_source, batch_size=batch_size, n_samples=n_samples,
                               color_format='yuv')
it = iter(data_generator.generate())

augmenter = AugmenterEnsemble(augmenters=[(0.5, TransformDistort())])

set_writer = SetFileParser(sample_path, img_format='jpg', label_format='pkl')

n_batches = int(np.ceil(n_samples / batch_size))
for i in range(n_batches):
    tic()
    batch = next(it)
    imgs_aug = []
    labels_aug = []
    for img, label, img_path in batch:
        img_aug, label_aug = augmenter.transform(img, label)
        # show(img_aug.bgr,labels=label_aug)
        imgs_aug.append(img_aug)
        labels_aug.append(label_aug)
    set_writer.write(imgs_aug, labels_aug)
    toc("Batch {}/{} written after ".format(i, n_batches))
