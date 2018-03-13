import numpy as np

from samplegen.imggen.RandomImgGen import RandomImgGen
from samplegen.shotgen.ShotLoad import ShotLoad
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.timing import tic, toc, tuc
from utils.workdir import work_dir

work_dir()

background_path = ["resource/ext/backgrounds/google-fence-gate-industry/",
                   'resource/ext/backgrounds/lmsun/']
# background_path = ["resource/backgrounds/google-fence-gate-industry/"]

# background_path = "samplegen/resource/backgrounds/single"
# background_path = "samplegen/resource/backgrounds/single/"
# sample_path = "resource/samples/single_background_test/"
sample_path = "resource/ext/samples/bebop20k/"
shot_path = "resource/ext/shots/mult_gate_aligned/"

n_backgrounds = 200
batch_size = 100
n_batches = int(np.ceil(n_backgrounds / batch_size))
create_dirs([sample_path])
tic()

shot_loader = ShotLoad(shot_path, img_format='bmp')
set_writer = SetFileParser(sample_path, img_format='jpg', label_format='xml')

augmenter = None  # AugmenterEnsemble(augmenters=[(1.0, AugmenterResize((80, 166)))])
generator = RandomImgGen(background_path,
                         output_shape=(80, 166),
                         image_transformer=augmenter)
for i in range(n_batches):
    tic()
    shots, shot_labels = shot_loader.get_shots(n_shots=250)
    samples, labels = generator.generate(shots, shot_labels, n_backgrounds=batch_size)

    set_writer.write(samples, labels)
    toc("Batch: {1:d}/{2:d} - {0:d}/{3:d} samples generated ".format(
        int(len(samples) * (i + 1)), i, n_batches, int(n_batches * batch_size)))
    tuc("Total time: ")

toc("Finished. Total run time ")
