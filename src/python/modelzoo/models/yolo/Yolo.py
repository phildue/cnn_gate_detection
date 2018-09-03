import numpy as np

from modelzoo.backend.tensor.yolo.ShallowYolo import ShallowYolo
from modelzoo.backend.tensor.yolo.ThinYolo import ThinYolo
from modelzoo.backend.tensor.yolo.TinyYolo import TinyYolo
from modelzoo.backend.tensor.yolo.YoloBase import YoloBase
from modelzoo.backend.tensor.yolo.YoloLoss import YoloLoss
from modelzoo.backend.tensor.yolo.YoloV2 import YoloV2
from modelzoo.models.Postprocessor import Postprocessor
from modelzoo.models.Predictor import Predictor
from modelzoo.models.Preprocessor import Preprocessor
from modelzoo.models.yolo.YoloDecoder import YoloDecoder
from modelzoo.models.yolo.YoloEncoder import YoloEncoder
from utils.imageprocessing.transform.YoloAugmenter import YoloAugmenter
from utils.labels.ObjectLabel import ObjectLabel


class Yolo(Predictor):
    @property
    def input_shape(self):
        return self.norm[0], self.norm[1], 3

    @staticmethod
    def yolo_v2(norm=(416, 416),
                grid=(13, 13),
                anchors=None,
                batch_size=8,
                scale_noob=0.5,
                scale_conf=5.0,
                scale_coor=5.0,
                scale_prob=1.0,
                conf_thresh=0.3,
                class_names=None,
                weight_file=None,
                color_format='yuv', augmenter=None):

        if anchors is None:
            anchors = np.array([[1.3221, 1.73145],
                                [3.19275, 4.00944],
                                [5.05587, 8.09892],
                                [9.47112, 4.84053],
                                [11.2364, 10.0071]])

        if class_names is None:
            class_names = [
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]

        ObjectLabel.classes = class_names

        n_boxes = anchors.shape[0]
        loss = YoloLoss(n_boxes=n_boxes,
                        n_classes=len(class_names),
                        weight_loc=scale_coor,
                        weight_conf=scale_conf,
                        weight_prob=scale_prob,
                        weight_noobj=scale_noob)
        net = YoloV2(loss=loss,
                     norm=norm,
                     grid=grid,
                     n_classes=len(class_names),
                     weight_file=weight_file,
                     anchors=anchors,
                     n_boxes=n_boxes)

        return Yolo(net,
                    class_names=class_names,
                    anchors=anchors,
                    batch_size=batch_size,
                    grid=grid,
                    norm=norm,
                    conf_thresh=conf_thresh,
                    color_format=color_format, augmenter=augmenter)

    @staticmethod
    def create_by_arch(
            class_names,
            color_format,
            architecture,
            augmenter,
            anchors,
            norm=(416, 416),
            batch_size=8,
            scale_noob=0.5,
            scale_conf=5.0,
            scale_coor=5.0,
            scale_prob=1.0,
            conf_thresh=0.3,
            weight_file=None,
    ):

        n_boxes = [len(a) for a in anchors]

        loss = YoloLoss(n_boxes=n_boxes,
                        n_classes=len(class_names),
                        weight_loc=scale_coor,
                        weight_conf=scale_conf,
                        weight_prob=scale_prob,
                        weight_noobj=scale_noob)
        net = YoloBase(
            anchors=anchors,
            architecture=architecture,
            loss=loss,
            n_boxes=n_boxes,
            img_shape=norm,
            weight_file=weight_file,
            n_classes=len(class_names))

        return Yolo(net,
                    augmenter=augmenter,
                    class_names=class_names,
                    anchors=anchors,
                    batch_size=batch_size,
                    grid=net.grid,
                    norm=norm,
                    conf_thresh=conf_thresh,
                    color_format=color_format)

    def __init__(self,
                 net,
                 augmenter,
                 grid,
                 color_format,
                 anchors,
                 class_names,
                 norm=(416, 416),
                 batch_size=8,
                 conf_thresh=0.3,
                 iou_thresh=0.4):

        self.color_format = color_format

        ObjectLabel.classes = class_names

        self.n_classes = len(class_names)
        self.anchors = anchors
        self.batch_size = batch_size
        self.grid = grid
        self.norm = norm
        self.conf_thresh = conf_thresh
        self.n_boxes = anchors.shape[0]

        encoder = YoloEncoder(
            anchor_dims=anchors,
            img_norm=norm,
            grids=grid,
            n_boxes=self.n_boxes)
        preprocessor = Preprocessor(augmenter=augmenter,
                                    encoder=encoder,
                                    img_shape=self.norm,
                                    n_classes=self.n_classes + 1,
                                    color_format=color_format)

        decoder = YoloDecoder(n_classes=len(class_names),
                              norm=norm,
                              grid=grid)
        postprocessor = Postprocessor(decoder=decoder,
                                      conf_thresh=self.conf_thresh,
                                      iou_thresh=iou_thresh)

        super().__init__(preprocessor=preprocessor,
                         postprocessor=postprocessor,
                         net=net,
                         loss=net.loss,
                         encoder=encoder,
                         decoder=decoder)
