import keras.backend as K
import numpy as np

from modelzoo.models.gatenet.AveragePrecisionGateNet import AveragePrecisionGateNet
from modelzoo.models.gatenet.DetectionCountGateNet import DetectionCountGateNet
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
ap = AveragePrecisionGateNet(encoder.n_boxes, grids, iou_thresh=0.6, norm=norm, batch_size=1)
dc = DetectionCountGateNet(encoder.n_boxes, grids, iou_thresh=0.6, norm=norm, batch_size=1)
y_true = K.placeholder(shape=[1, 13 * 13 * 3 + 26 * 26 * 3, 5 + 6], dtype=K.tf.float64)
y_pred = K.placeholder(shape=[1, 13 * 13 * 3 + 26 * 26 * 3, 5 + 6], dtype=K.tf.float64)
ap_graph = ap.compute(y_true, y_pred)
dc_graph = dc.compute(y_true, y_pred)
tp_graph = ap.total_precision(y_true,y_pred)
batch = next(dataset)

# label_t = encoder.encode_label(ImgLabel([]))
# step = 500
# for i in range(label_t.shape[0]-step):
#     label_dec = decoder.decode_netout(label_t[i:i+step])
#     show(Image(np.zeros(norm), 'bgr'), labels=[label_dec], colors=[COLOR_BLUE], name='Anchors')

# sess = K.tf.InteractiveSession()
with K.get_session() as sess:
    sess.run(K.tf.global_variables_initializer())
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

        ap_out = sess.run(ap_graph, {y_true: np.expand_dims(label_t, 0),
                                     y_pred: np.expand_dims(label_t, 0)})

        dc_out = sess.run(dc_graph, {y_true: np.expand_dims(label_t, 0),
                                     y_pred: np.expand_dims(label_t, 0)})

        tp_out = sess.run(tp_graph, {y_true: np.expand_dims(label_t, 0),
                                     y_pred: np.expand_dims(label_t, 0)})

        l_loc_numpy = np.sum(np.sum((label_t[:, 1:5] - label_next_t[:, 1:5]) ** 2, -1) * loss.scale_coor)/1
        print('L loc=', l_loc)
        print('Total Loc loss', l_loc_numpy)
        print('L conf=', l_conf)
        print('AP: ', ap_out)
        print('DC: ', dc_out)
        print('TP:',tp_out)
        label_dec = decoder.decode_netout(label_t)
        label_dec.objects = [o for o in label_dec.objects if o.confidence > 0]

        show(img, labels=[label], colors=[COLOR_GREEN], name='True')
        show(img, labels=[label_dec], colors=[COLOR_RED], name='Decoded')
        show(img, labels=[label, label_next], colors=[COLOR_GREEN, COLOR_BLUE], name='Diff')
