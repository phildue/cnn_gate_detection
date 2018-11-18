from pprint import pprint

from modelzoo.Decoder import Decoder
from modelzoo.Encoder import Encoder
from modelzoo.Postprocessor import Postprocessor
from modelzoo.Preprocessor import Preprocessor
from modelzoo.build_model import build_detector
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Imageprocessing import show, LEGEND_POSITION
from utils.imageprocessing.transform.TransformCrop import TransformCrop
from utils.imageprocessing.transform.TransformResize import TransformResize
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

cd_work()
# 'resource/ext/samples/iros2018_course_final_simple_17gates'
generator = GateGenerator(directories=['resource/ext/samples/iros2018_course_final_simple_17gates'],
                          batch_size=8, color_format='bgr',
                          shuffle=False, start_idx=0, valid_frac=0,
                          label_format='xml',
                          img_format='jpg'
                          )
#
# generator = VocGenerator(batch_size=8)
#
# model = SSD.ssd300(n_classes=20, conf_thresh=0.1, color_format='bgr', weight_file='logs/ssd300_voc3/SSD300.h5',
#                    iou_thresh_nms=0.3)
# model = Yolo.yolo_v2(class_names=['gate'], batch_size=8, conf_thresh=0.5,
#                      color_format='yuv', weight_file='logs/v2_mixed/model.h5')
# model = Yolo.tiny_yolo(class_names=['gate'], batch_size=8, conf_thresh=0.5,
#                        color_format='yuv', weight_file='logs/tiny_mixed/model.h5')
src_dir = 'out/mavnet_lowres320/'
summary = load_file(src_dir + 'summary.pkl')
pprint(summary['architecture'])
iou_thresh = 0.6
preprocessing = [TransformCrop(0, 52, 416, 416 - 52), TransformResize((240, 320))]
model, output_grids = build_detector(img_shape=(summary['img_res'][0], summary['img_res'][1], 3),
                                     architecture=summary['architecture'],
                                     anchors=summary['anchors'],
                                     n_polygon=4)
model.load_weights(src_dir + '/model.h5')
encoder = Encoder(anchor_dims=summary['anchors'], img_norm=summary['img_res'], grids=output_grids, n_polygon=4,
                  iou_min=0.4)
preprocessor = Preprocessor(preprocessing=preprocessing, encoder=encoder, n_classes=1, img_shape=summary['img_res'],
                            color_format='bgr')
decoder = Decoder(anchor_dims=summary['anchors'], n_polygon=4, norm=summary['img_res'], grid=output_grids)
postproessor = Postprocessor(decoder=decoder)
# _model = build_detector((480, 640, 3), architecture=summary['architecture'], anchors=summary['anchors'])
# _model.load_weights(src_dir + '/model.h5')
# model.net.backend = _model
# create_dirs(['out/1807/narrow_strides_late_bottleneck416x416-13x13+9layers/img04/'])
n_samples = generator.n_samples
iterator = iter(generator.generate())
idx = 0
for i in range(int(n_samples / generator.batch_size)):
    batch = next(iterator)
    for i in range(len(batch)):

        img = batch[i][0]
        label = batch[i][1]
        # show(img, 'demo',
        #      colors=[(255, 255, 255), (0, 0, 255), (255, 0, 0)],
        #      legend=LEGEND_TEXT, t=t_show)

        l = postproessor.postprocess(model.predict(preprocessor.preprocess(img)))

        l.objects = [o for o in l.objects if o.confidence > 0.5]
        if preprocessing:
            for p in preprocessing:
                img, label = p.transform(img, label)

        boxes_pred = l.objects
        boxes_true = label.objects

        false_negatives = []
        false_positives = boxes_pred.copy()
        true_positives = []
        for j in range(len(boxes_true)):
            match = False
            box_true = boxes_true[j]
            for k in range(len(false_positives)):
                box_pred = false_positives[k]
                match = box_pred.poly.iou(box_true.poly) > iou_thresh and box_pred.class_id == box_true.class_id
                if match:
                    true_positives.append(box_pred)
                    false_positives.remove(box_pred)
                    break
            if not match:
                false_negatives.append(box_true)

        label_tp = ImgLabel(true_positives)
        label_fp = ImgLabel(false_positives)
        label_fn = ImgLabel(false_negatives)

        show(img, 'demo', labels=[label_tp, label_fp, label],
             colors=[(255, 255, 255), (0, 0, 255), (255, 0, 0)],
             legend=LEGEND_POSITION, t=0)

        # save_labeled(img.bgr, out_file + '/{0:04d}.jpg'.format(idx), labels=[label_tp, label_fp, label],
        #              colors=[(255, 255, 255), (0, 0, 255), (255, 0, 0)],
        #              legend=LEGEND_TEXT)
        idx += 1
