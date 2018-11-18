import numpy as np
from modelzoo.GateNetDecoder import GateNetDecoder
from modelzoo.GateNetEncoder import Encoder

from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show
from utils.workdir import cd_work

cd_work()
batch_size = 10
anchor = np.array([[
    [330, 340],
    [235, 240],
    [160, 165]],
    [[25, 40],
     [65, 70],
     [100, 110]]])
norm = (416, 416)
grids = [(13, 13), (26, 26)]
encoder = Encoder(anchor_dims=anchor, img_norm=norm, grids=grids, verbose=True)

decoder = GateNetDecoder(anchor_dims=anchor, norm=norm, grid=grids)

dataset = GateGenerator(["resource/ext/samples/iros2018_course_final_simple_17gates/"], batch_size=batch_size,
                        color_format='bgr', label_format='xml', n_samples=99).generate()
batch = next(dataset)

# label_t = encoder.encode_label(ImgLabel([]))
# step = 500
# for i in range(label_t.shape[0]-step):
#     label_dec = decoder.decode_netout(label_t[i:i+step])
#     show(Image(np.zeros(norm), 'bgr'), labels=[label_dec], colors=[COLOR_BLUE], name='Anchors')

for img, label, _ in batch:
    print(label)
    print('_____________________________')
    label_t = encoder.encode_label(label)
    print(label_t)
    print('_____________________________')

    assigned = label_t[label_t[:, 0] > 0]
    # n_assigned = len(assigned)
    print('Assigned: {}'.format(assigned))
    # print("N Assigned: {}".format(n_assigned))

    label_dec = decoder.decode_netout(label_t)
    label_dec.objects = [o for o in label_dec.objects if o.confidence > 0]
    show(img, labels=[label], colors=[COLOR_GREEN], name='True')
    show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded')
