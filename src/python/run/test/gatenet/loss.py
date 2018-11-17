import keras.backend as K
import numpy as np

from modelzoo.models.gatenet.AveragePrecisionGateNet import AveragePrecisionGateNet
from modelzoo.models.gatenet.GateDetectionLoss import GateDetectionLoss
from modelzoo.models.gatenet.GateNetDecoder import GateNetDecoder
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Imageprocessing import COLOR_GREEN, COLOR_RED, show, COLOR_BLUE
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
encoder = GateNetEncoder(anchor_dims=anchor, img_norm=norm, grids=grids)

decoder = GateNetDecoder(anchor_dims=anchor, norm=norm, grid=grids)

dataset = GateGenerator(["resource/ext/samples/iros2018_course_final_simple_17gates/"], batch_size=batch_size,
                        color_format='bgr', label_format='xml', n_samples=99).generate()
loss = GateDetectionLoss()
mAP = AveragePrecisionGateNet(encoder.n_boxes, grids, iou_thresh=0.6, norm=norm, batch_size=1)
batch = next(dataset)

# label_t = encoder.encode_label(ImgLabel([]))
# step = 500
# for i in range(label_t.shape[0]-step):
#     label_dec = decoder.decode_netout(label_t[i:i+step])
#     show(Image(np.zeros(norm), 'bgr'), labels=[label_dec], colors=[COLOR_BLUE], name='Anchors')
sess = K.tf.InteractiveSession()
for i in range(len(batch)):
    img = batch[i][0]
    label = batch[i][1]
    img_next = batch[i + 1][0]
    label_next = batch[i + 1][1]

    print(label)
    print('_____________________________')
    label_t = encoder.encode_label(label)
    label_next_t = encoder.encode_label(label_next)
    print(label_t)
    print('_____________________________')

    assigned = label_t[label_t[:, 0] > 0]
    # n_assigned = len(assigned)
    print('Assigned: {}'.format(assigned))
    # print("N Assigned: {}".format(n_assigned))

    l_loc = loss.localization_loss(K.expand_dims(label_t, 0), K.expand_dims(label_next_t, 0)).eval()
    l_conf = loss.confidence_loss(K.expand_dims(label_t, 0), K.expand_dims(label_next_t, 0)).eval()
    # m = mAP.compute(K.expand_dims(label_t, 0), K.expand_dims(label_next_t, 0)).eval()
    print('L loc=', l_loc)
    print('L conf=', l_conf)
    # print('AP: '.format(l))
    label_dec = decoder.decode_netout(label_t)
    label_dec.objects = [o for o in label_dec.objects if o.confidence > 0]

    show(img, labels=[label], colors=[COLOR_GREEN], name='True')
    show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded')
    show(img, labels=[label, label_next], colors=[COLOR_GREEN, COLOR_BLUE], name='Diff')
