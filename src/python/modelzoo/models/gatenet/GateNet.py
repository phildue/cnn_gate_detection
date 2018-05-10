import numpy as np

from modelzoo.backend.tensor.gatenet.GateDetectionLoss import GateDetectionLoss
from modelzoo.backend.tensor.gatenet.GateNetV0 import GateNetV0
from modelzoo.backend.tensor.gatenet.GateNetV1 import GateNetV1
from modelzoo.backend.tensor.gatenet.GateNetV10 import GateNetV10
from modelzoo.backend.tensor.gatenet.GateNetV2 import GateNetV2
from modelzoo.backend.tensor.gatenet.GateNetV3 import GateNetV3
from modelzoo.backend.tensor.gatenet.GateNetV4 import GateNetV4
from modelzoo.backend.tensor.gatenet.GateNetV5 import GateNetV5
from modelzoo.backend.tensor.gatenet.GateNetV6 import GateNetV6
from modelzoo.backend.tensor.gatenet.GateNetV7 import GateNetV7
from modelzoo.backend.tensor.gatenet.GateNetV8 import GateNetV8
from modelzoo.backend.tensor.gatenet.GateNetV9 import GateNetV9
from modelzoo.models.Postprocessor import Postprocessor
from modelzoo.models.Predictor import Predictor
from modelzoo.models.Preprocessor import Preprocessor
from modelzoo.models.gatenet.GateNetDecoder import GateNetDecoder
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ObjectLabel import ObjectLabel


class GateNet(Predictor):
    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self.norm[0], self.norm[1], 3

    @staticmethod
    def v0(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV0(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v1(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV1(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v2(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV2(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v3(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV3(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v4(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV4(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v5(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV5(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v6(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV6(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v7(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV7(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v8(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV8(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v9(norm=(416, 416),
           grid=(13, 13),
           anchors=None,
           batch_size=8,
           scale_noob=1.0,
           scale_conf=5.0,
           scale_coor=1.0,
           scale_prob=1.0,
           conf_thresh=0.3,
           weight_file=None,
           color_format='yuv',
           augmenter: ImgTransform = None,
           n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV9(loss=loss,
                        anchors=anchors,
                        img_shape=norm,
                        grid=grid,
                        weight_file=weight_file,
                        n_boxes=n_boxes,
                        n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)

    @staticmethod
    def v10(norm=(416, 416),
            grid=(13, 13),
            anchors=None,
            batch_size=8,
            scale_noob=1.0,
            scale_conf=5.0,
            scale_coor=1.0,
            scale_prob=1.0,
            conf_thresh=0.3,
            weight_file=None,
            color_format='yuv',
            augmenter: ImgTransform = None,
            n_polygon=4):

        if anchors is None:
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])

        n_boxes = anchors.shape[0]
        loss = GateDetectionLoss(grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)

        net = GateNetV10(loss=loss,
                         anchors=anchors,
                         img_shape=norm,
                         grid=grid,
                         weight_file=weight_file,
                         n_boxes=n_boxes,
                         n_polygon=n_polygon)

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon)


    def __init__(self,
                 net,
                 norm=(416, 416),
                 grid=(13, 13),
                 anchors=None,
                 batch_size=8,
                 conf_thresh=0.3,
                 color_format='yuv',
                 iou_thresh=0.4,
                 augmenter: ImgTransform = None,
                 n_polygon=4):

        self.color_format = color_format
        if anchors is None:
            anchors = np.array([[1.3221, 1.73145],
                                [3.19275, 4.00944],
                                [5.05587, 8.09892],
                                [9.47112, 4.84053],
                                [11.2364, 10.0071]])

        ObjectLabel.classes = ['gate']

        self.anchors = anchors
        self.batch_size = batch_size
        self.grid = grid
        self.norm = norm
        self.conf_thresh = conf_thresh
        self.n_boxes = anchors.shape[0]
        self._output_shape = grid[0] * grid[1], self.n_boxes * (n_polygon + 1)

        encoder = GateNetEncoder(img_norm=norm,
                                 grid=grid,
                                 n_boxes=self.n_boxes,
                                 n_polygon=n_polygon)
        preprocessor = Preprocessor(augmenter=augmenter,
                                    encoder=encoder,
                                    img_shape=self.norm,
                                    n_classes=1,
                                    color_format=color_format)

        decoder = GateNetDecoder(norm=norm,
                                 grid=grid,
                                 n_polygon=n_polygon)
        postprocessor = Postprocessor(decoder=decoder,
                                      conf_thresh=self.conf_thresh,
                                      iou_thresh=iou_thresh)

        super().__init__(preprocessor=preprocessor,
                         postprocessor=postprocessor,
                         net=net,
                         loss=net.loss,
                         encoder=encoder,
                         decoder=decoder)
