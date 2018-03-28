import numpy as np

from samplegen.imggen.DetermImgGen import DetermImgGen
from samplegen.shotgen.ShotLoad import ShotLoad
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.timing import tic, toc, tuc
from utils.workdir import cd_work

cd_work()

# background_path = ["samplegen/resource/backgrounds/lmsun/",
#                    "samplegen/resource/backgrounds/google-fence-gate-industry/"]
# background_path = "samplegen/resource/backgrounds/single"
background_path = "resource/backgrounds/single/"
# sample_path = "resource/samples/single_background_test/"
sample_path = "resource/samples/stream_valid3/"
shot_path = "resource/shots/stream3/"

n_backgrounds = 1
batch_size = 1
n_batches = int(np.ceil(n_backgrounds / batch_size))

create_dirs([sample_path])
tic()

setwriter = SetFileParser(sample_path, img_format='jpg', label_format='pkl')

for i in range(n_batches):
    tic()
    shots, shot_labels = ShotLoad(shot_path, img_format='bmp', random=False).get_shots(n_shots=600)
    samples, labels = DetermImgGen(background_path, out_format='bgr').generate(
        shots, shot_labels,
        n_backgrounds=n_backgrounds)

    setwriter.write(samples, labels)
    toc("Batch: {1:d}/{2:d} - {0:d}/{3:d} samples generated ".format(
        int(len(samples) * (i + 1)), i, n_batches, int(n_batches * batch_size)))
    tuc("Total time: ")

toc("Finished. Total run time ")
