import numpy as np
from backend.tensor.yolo.TinyYolo import TinyYolo
from backend.tensor.yolo.YoloLoss import YoloLoss
from frontend.augmentation.YoloAugmenter import YoloAugmenter
from frontend.models.Postprocessor import Postprocessor
from frontend.models.Predictor import Predictor
from frontend.models.Preprocessor import Preprocessor
from frontend.models.yolo.YoloDecoder import YoloDecoder

from src.python.modelzoo.backend.tensor.yolo.YoloV2 import YoloV2
from src.python.modelzoo.models.yolo.YoloEncoder import YoloEncoder
from src.python.utils.labels import ObjectLabel


class Yolo(Predictor):
    @property
    def input_shape(self):
        return self.norm[0], self.norm[1], 3

    @staticmethod
    def tiny_yolo(norm=(416, 416),
                  grid=(13, 13),
                  anchors=None,
                  batch_size=8,
                  scale_noob=1.0,
                  scale_conf=5.0,
                  scale_coor=1.0,
                  scale_prob=1.0,
                  conf_thresh=0.3,
                  class_names=None,
                  weight_file=None,
                  color_format='yuv'):

        if anchors is None:
            anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

        if class_names is None:
            class_names = [
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]

        n_boxes = int(np.floor(len(anchors) / 2))
        loss = YoloLoss(grid=grid,
                        n_boxes=n_boxes,
                        n_classes=len(class_names),
                        verbose=False,
                        weight_loc=scale_coor,
                        weight_conf=scale_conf,
                        weight_prob=scale_prob,
                        weight_noobj=scale_noob)

        net = TinyYolo(loss=loss,
                       anchors=anchors,
                       norm=norm,
                       grid=grid,
                       n_classes=len(class_names),
                       weight_file=weight_file)

        return Yolo(net,
                    class_names=class_names,
                    anchors=anchors,
                    batch_size=batch_size,
                    grid=grid,
                    norm=norm,
                    conf_thresh=conf_thresh,
                    color_format=color_format)

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
                color_format='yuv'):

        if anchors is None:
            anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

        if class_names is None:
            class_names = [
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]

        ObjectLabel.classes = class_names

        n_boxes = int(np.floor(len(anchors) / 2))
        loss = YoloLoss(grid=grid,
                        n_boxes=n_boxes,
                        n_classes=len(class_names),
                        verbose=False,
                        weight_loc=scale_coor,
                        weight_conf=scale_conf,
                        weight_prob=scale_prob,
                        weight_noobj=scale_noob)
        net = YoloV2(loss=loss,
                     norm=norm,
                     grid=grid,
                     n_classes=len(class_names),
                     weight_file=weight_file,
                     anchors=anchors)

        return Yolo(net,
                    class_names=class_names,
                    anchors=anchors,
                    batch_size=batch_size,
                    grid=grid,
                    norm=norm,
                    conf_thresh=conf_thresh,
                    color_format=color_format)

    def __init__(self,
                 net,
                 norm=(416, 416),
                 grid=(13, 13),
                 anchors=None,
                 batch_size=8,
                 conf_thresh=0.3,
                 class_names=None,
                 color_format='yuv', iou_thresh=0.4):

        self.color_format = color_format
        if anchors is None:
            anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
        if class_names is None:
            class_names = [
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]

        ObjectLabel.classes = class_names

        self.n_classes = len(class_names)
        self.anchors = anchors
        self.batch_size = batch_size
        self.grid = grid
        self.norm = norm
        self.conf_thresh = conf_thresh
        self.n_boxes = int(np.floor(len(anchors) / 2))

        encoder = YoloEncoder(img_norm=norm,
                              grid=grid,
                              n_boxes=self.n_boxes,
                              n_classes=self.n_classes)
        preprocessor = Preprocessor(augmenter=YoloAugmenter(),
                                    encoder=encoder,
                                    img_shape=self.norm,
                                    n_classes=self.n_classes + 1,
                                    color_format=color_format)

        decoder = YoloDecoder(norm=norm,
                              grid=grid,
                              class_names=class_names)
        postprocessor = Postprocessor(decoder=decoder,
                                      conf_thresh=self.conf_thresh,
                                      iou_thresh=iou_thresh)

        super().__init__(preprocessor=preprocessor,
                         postprocessor=postprocessor,
                         net=net,
                         loss=net.loss,
                         encoder=encoder,
                         decoder=decoder)
