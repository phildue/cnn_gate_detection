import numpy as np

from samplegen.imggen.RandomImgGen import RandomImgGen
from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from samplegen.shotgen.ShotLoad import ShotLoad
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.transform.RandomBlur import RandomBlur
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomGrayNoise import RandomGrayNoise
from utils.imageprocessing.transform.TransformBrightness import TransformBrightness
from utils.imageprocessing.transform.TransformDistort import TransformDistort
from utils.imageprocessing.transform.TransformGrayNoise import TransformGrayNoise
from utils.imageprocessing.transform.TransformResize import TransformResize
from utils.timing import tic, toc, tuc
from utils.workdir import work_dir

work_dir()

background_path = ["resource/ext/backgrounds/google-fence-gate-industry/",
                   'resource/ext/backgrounds/lmsun/']
# background_path = ["resource/backgrounds/google-fence-gate-industry/"]

# background_path = "samplegen/resource/backgrounds/single"
# background_path = "samplegen/resource/backgrounds/single/"
# sample_path = "resource/samples/single_background_test/"
sample_path = "resource/ext/samples/bebop_merge/"
shot_path = "resource/ext/shots/thick_square/"

n_backgrounds = 100
batch_size = 100
output_shape = (160, 315)
n_batches = int(np.ceil(n_backgrounds / batch_size))
create_dirs([sample_path])
tic()

shot_loader = ShotLoad(shot_path, img_format='bmp')
set_writer = SetFileParser(sample_path, img_format='jpg', label_format='xml', start_idx=20000)

augmenter = None  # RandomEnsemble(augmenters=[
# (1.0, TransformResize((160, 315))),
# (1.0, TransformDistort(BarrelDistortion.from_file('resource/distortion_model_est.pkl'), crop=0.1))])

shot_augmenter = RandomEnsemble([
    (1.0, TransformBrightness(0.5)),
    (1.0, TransformGrayNoise(0, 10)),
    (1.0, RandomBlur((5, 5), 0, 10))
])
generator = RandomImgGen(background_path,
                         output_shape=output_shape,
                         image_transformer=augmenter)
for i in range(n_batches):
    tic()
    shots = shot_loader.get_shots(n_shots=250)

    shots_img, shot_labels = [shot_augmenter.transform(s, l) for s, l in shots]

    samples, labels = generator.generate(shots_img, shot_labels, n_backgrounds=batch_size)

    set_writer.write(samples, labels)
    toc("Batch: {1:d}/{2:d} - {0:d}/{3:d} samples generated ".format(
        int(len(samples) * (i + 1)), i, n_batches, int(n_batches * batch_size)))
    tuc("Total time: ")

toc("Finished. Total run time ")

SetAnalyzer(output_shape, shot_path).show_summary()
