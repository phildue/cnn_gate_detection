from fileaccess.VocGenerator import VocGenerator

from models.ssd.SSD import SSD

from workdir import work_dir

work_dir()

from visualization.utils import demo_generator

# generator = GateGenerator(directory='resource/samples/stream_valid1/', batch_size=8, color_format='bgr',
#                          shuffle=False, start_idx=400)


generator = VocGenerator("resource/backgrounds/VOCdevkit/VOC2012/Annotations/",
                         "resource/backgrounds/VOCdevkit/VOC2012/JPEGImages/", batch_size=8)

model = SSD.ssd7(n_classes=20, weight_file='logs/ssd7/SSD7.h5', conf_thresh=0.7)

demo_generator(model, generator, t_show=0)
