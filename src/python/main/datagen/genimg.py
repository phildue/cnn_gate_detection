import numpy as np

from samplegen.imggen.RandomImgGen import RandomImgGen
from samplegen.shotgen.ShotLoad import ShotLoad
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.timing import tic, toc, tuc
from utils.workdir import work_dir

work_dir()

background_path = ["resource/backgrounds/lmsun/",
                   "resource/backgrounds/google-fence-gate-industry/"]
# background_path = ["resource/backgrounds/google-fence-gate-industry/"]

# background_path = "samplegen/resource/backgrounds/single"
# background_path = "samplegen/resource/backgrounds/single/"
# sample_path = "resource/samples/single_background_test/"
sample_path = "resource/samples/mult_gate_aligned/"
shot_path = "resource/shots/mult_gate_aligned/"

n_backgrounds = 20000
batch_size = 100
n_batches = int(np.ceil(n_backgrounds / batch_size))
create_dirs([sample_path])
tic()

set_writer = SetFileParser(sample_path, img_format='jpg', label_format='xml')

for i in range(n_batches):
    tic()
    shots, shot_labels = ShotLoad(shot_path, img_format='bmp').get_shots(n_shots=250)
    samples, labels = RandomImgGen(background_path, out_format='yuv',
                                   blur_kernel=(5, 5),
                                   blur_it=20,
                                   noisy_it=10,
                                   noisy_var=30.0).generate(
        shots, shot_labels,
        n_backgrounds=batch_size)

    set_writer.write(samples, labels)
    toc("Batch: {1:d}/{2:d} - {0:d}/{3:d} samples generated ".format(
        int(len(samples) * (i + 1)), i, n_batches, int(n_batches * batch_size)))
    tuc("Total time: ")

toc("Finished. Total run time ")
