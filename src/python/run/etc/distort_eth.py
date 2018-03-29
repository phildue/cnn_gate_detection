from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.BarrelDistortion import BarrelDistortion
from utils.imageprocessing.transform.TransformDistort import TransformDistort
from utils.workdir import cd_work

image_source = ['resource/samples/video/eth']
cd_work()
generator = GateGenerator(directories=image_source, batch_size=150, img_format='jpg',
                          shuffle=False, color_format='bgr', label_format='pkl',start_idx=0)


batch = next(iter(generator.generate()))

create_dirs(['resource/samples/video/eth_distorted'])
set_writer = SetFileParser('resource/samples/video/eth_distorted', img_format='jpg', label_format='xml', start_idx=0)

dist_model = TransformDistort(BarrelDistortion.from_file('resource/distortion_model_est.pkl'))

img_dist = []
label_dist=[]
for i in range(len(batch)):
    img, label, image_file = batch[i]
    img,label = resize(img,(180,315),label=label)
    img_d, label_d = dist_model.transform(img,label)
    img_dist.append(img_d)
    label_dist.append(label_d)

set_writer.write(img_dist,label_dist)