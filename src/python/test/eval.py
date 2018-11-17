import keras.backend as K
import numpy as np
from utils.fileaccess.CropGenerator import CropGenerator

from modelzoo.models.gatenet import AveragePrecisionGateNet
from modelzoo.models.gatenet.GateNet import GateNet
from modelzoo.models.gatenet.PrecisionRecallGateNet import PrecisionRecallGateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import COLOR_BLUE, COLOR_GREEN, show
from utils.workdir import cd_work

cd_work()
# Image Source
batch_size = 8
image_source = ['resource/ext/samples/industrial_new_test/']
color_format = 'bgr'

# Model
model_src = 'out/refnet52x52->3x3+4layers+32filters/'
summary = load_file(model_src + 'summary.pkl')
architecture = summary['architecture']
img_res = 52, 52
grid = 3, 3
predictor = GateNet.create_by_arch(norm=img_res, architecture=architecture,
                                   anchors=np.array([[[1, 1],
                                                      [1 / grid[0], 1 / grid[1]],  # img_h/img_w
                                                      [2 / grid[0], 2 / grid[1]],  # 0.5 img_h/ 0.5 img_w
                                                      [1 / grid[0], 3 / grid[0]],  # img_h / 0.33 img_w
                                                      [1 / grid[0], 2 / grid[0]]  # img_h / 0.5 img_w
                                                      ]]),
                                   batch_size=batch_size,
                                   color_format='yuv',
                                   augmenter=None,
                                   weight_file=model_src + 'model.h5'
                                   )

generator = CropGenerator(
    GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg', valid_frac=0.0,
                  shuffle=True, color_format=color_format, label_format='xml', start_idx=0))

# Evaluator
iou_thresh = 0.4
precision_recall = PrecisionRecallGateNet(n_boxes=predictor.n_boxes, grid=predictor.grid, iou_thresh=iou_thresh,
                                          norm=img_res,
                                          batch_size=8)

average_precision = AveragePrecisionGateNet(n_boxes=predictor.n_boxes, grid=predictor.grid, iou_thresh=iou_thresh,
                                            norm=img_res,
                                            batch_size=8)

iterator = iter(generator.generate())
y_pred = K.placeholder((batch_size, predictor.grid[0][0] * predictor.grid[0][1] * 5, 5 + 4),
                       name='y_pred')
y_true = K.placeholder((batch_size, predictor.grid[0][0] * predictor.grid[0][1] * 5, 5 + 4),
                       name='y_true')
pr = precision_recall.compute(
    y_true=y_true,
    y_pred=y_pred)
ap = average_precision.compute(y_true=y_true,
                               y_pred=y_pred)
with K.get_session() as sess:
    # sess.run(K.tf.global_variables_initializer())
    for i in range(int(generator.n_samples / batch_size)):
        batch = next(iterator)
        x, y_true_t = predictor.preprocessor.preprocess_test(batch)
        y_pred_t = predictor.net.predict(x)
        precision, recall, _ = sess.run(pr, {y_true: y_true_t,
                                             y_pred: y_pred_t})
        ap_batch = sess.run(ap, {y_true: y_true_t,
                                 y_pred: y_pred_t})
        print(ap_batch)
        for j in range(batch_size):
            img, label_true, _ = batch[j]
            label_pred = predictor.predict(img)
            img,label_true = resize(img,label=label_true,shape=predictor.input_shape)

            #label_pred = resize_label(label_pred, predictor.input_shape, img.shape)
            show(img, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_BLUE], t=1)
        # print(precision)
        # print(recall)
