from os.path import expanduser

import numpy as np

from samplegen.imggen.RandomImgGen import RandomImgGen
from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from samplegen.shotgen.ShotLoad import ShotLoad
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.transform.RandomBlur import RandomBlur
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomGrayNoise import RandomGrayNoise
from utils.imageprocessing.transform.RandomMerge import RandomMerge
from utils.imageprocessing.transform.RandomMotionBlur import RandomMotionBlur
from utils.imageprocessing.transform.TransformBlur import TransformBlur
from utils.imageprocessing.transform.TransformBrightness import TransformBrightness
from utils.imageprocessing.transform.TransformDistort import TransformDistort
from utils.imageprocessing.transform.TransformGrayNoise import TransformGrayNoise
from utils.timing import tic, toc, tuc
from utils.workdir import cd_work, home
import numpy as np
import scipy.stats as st

cd_work()

background_path = [home() + "/Documents/datasets/google_background/"]
# background_path = ["resource/backgrounds/google-fence-gate-industry/"]

# background_path = "samplegen/resource/backgrounds/single"
# background_path = "samplegen/resource/backgrounds/single/"
# sample_path = "resource/samples/single_background_test/"
sample_path = "resource/samples/google_merge_distort/"
shot_path = home() + "/Documents/datasets/thick_square_roll/"

n_backgrounds = 5000
batch_size = 100
output_shape = (160, 315)
n_batches = int(np.ceil(n_backgrounds / batch_size))
create_dirs([sample_path])
tic()

shot_loader = ShotLoad(shot_path, img_format='bmp')
set_writer = SetFileParser(sample_path, img_format='jpg', label_format='xml', start_idx=0)

preprocessor = RandomEnsemble([
    (1.0, RandomBrightness(0.5, 3.0))])

postprocessor = RandomEnsemble([
    (1.0, TransformGrayNoise()),
    (0.5, RandomMerge(pixel_frac=0.005, kernel_size=(9, 9))),
    (0.1, RandomMotionBlur()),
    (1.0, TransformDistort(BarrelDistortion.from_file('distortion_model_est.pkl')))])


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


gauss_kernel = gkern(7)

kernels = [gauss_kernel]

generator = RandomImgGen(background_path,
                         output_shape=output_shape,
                         preprocessor=preprocessor,
                         merge_kernels=kernels,
                         postprocessor=postprocessor)
for i in range(n_batches):
    tic()
    shots_img, shots_labels = shot_loader.get_shots(n_shots=250)

    samples, labels = generator.generate(shots_img, shots_labels, n_backgrounds=batch_size)

    set_writer.write(samples, labels)
    toc("Batch: {1:d}/{2:d} - {0:d}/{3:d} samples generated ".format(
        int(len(samples) * (i + 1)), i, n_batches, int(n_batches * batch_size)))
    tuc("Total time: ")

toc("Finished. Total run time ")

# SetAnalyzer(output_shape, shot_path).show_summary()
