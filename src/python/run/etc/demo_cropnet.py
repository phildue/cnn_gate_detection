import numpy as np

from modelzoo.models.cropnet import CropNet
from modelzoo.models.cropnet import CropNetBase
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work

cd_work()

generator = iter(GateGenerator(directories=['resource/ext/samples/daylight_flight'],
                               batch_size=8, color_format='bgr',
                               shuffle=True, start_idx=0, valid_frac=0,
                               label_format='xml',
                               ).generate())

summary = load_file('out/cropnet52x52-8layers-48filters/summary.pkl')
model = CropNet(net=CropNetBase(architecture=summary['architecture']))

batch = next(generator)
img_org = batch[0][0]
img = resize(img_org, (52, 52))
out = model.net.predict(np.expand_dims(img.yuv.array, 0))[0]
out = np.expand_dims(out, -1)
out = np.tile(out, (1, 1, 3))
img_label = Image(out, 'bgr')
img_label = resize(img_label, img_org.shape)
show(img_label, t=1, name='Label')
show(img_org)
