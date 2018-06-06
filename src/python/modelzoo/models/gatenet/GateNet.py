import numpy as np

from modelzoo.backend.tensor.gatenet.GateDetectionLoss import GateDetectionLoss
from modelzoo.backend.tensor.gatenet.GateNetV0 import GateNetV0
from modelzoo.backend.tensor.gatenet.GateNetV1 import GateNetV1
from modelzoo.backend.tensor.gatenet.GateNetV10 import GateNetV10
from modelzoo.backend.tensor.gatenet.GateNetV11 import GateNetV11
from modelzoo.backend.tensor.gatenet.GateNetV12 import GateNetV12
from modelzoo.backend.tensor.gatenet.GateNetV13 import GateNetV13
from modelzoo.backend.tensor.gatenet.GateNetV14 import GateNetV14
from modelzoo.backend.tensor.gatenet.GateNetV15 import GateNetV15
from modelzoo.backend.tensor.gatenet.GateNetV16 import GateNetV16
from modelzoo.backend.tensor.gatenet.GateNetV17 import GateNetV17
from modelzoo.backend.tensor.gatenet.GateNetV18 import GateNetV18
from modelzoo.backend.tensor.gatenet.GateNetV19 import GateNetV19
from modelzoo.backend.tensor.gatenet.GateNetV2 import GateNetV2
from modelzoo.backend.tensor.gatenet.GateNetV20 import GateNetV20
from modelzoo.backend.tensor.gatenet.GateNetV21 import GateNetV21
from modelzoo.backend.tensor.gatenet.GateNetV22 import GateNetV22
from modelzoo.backend.tensor.gatenet.GateNetV23 import GateNetV23
from modelzoo.backend.tensor.gatenet.GateNetV24 import GateNetV24
from modelzoo.backend.tensor.gatenet.GateNetV25 import GateNetV25
from modelzoo.backend.tensor.gatenet.GateNetV26 import GateNetV26
from modelzoo.backend.tensor.gatenet.GateNetV28 import GateNetV28
from modelzoo.backend.tensor.gatenet.GateNetV29 import GateNetV29
from modelzoo.backend.tensor.gatenet.GateNetV3 import GateNetV3
from modelzoo.backend.tensor.gatenet.GateNetV30 import GateNetV30
from modelzoo.backend.tensor.gatenet.GateNetV31 import GateNetV31
from modelzoo.backend.tensor.gatenet.GateNetV32 import GateNetV32
from modelzoo.backend.tensor.gatenet.GateNetV33 import GateNetV33
from modelzoo.backend.tensor.gatenet.GateNetV34 import GateNetV34
from modelzoo.backend.tensor.gatenet.GateNetV35 import GateNetV35
from modelzoo.backend.tensor.gatenet.GateNetV36 import GateNetV36
from modelzoo.backend.tensor.gatenet.GateNetV37 import GateNetV37
from modelzoo.backend.tensor.gatenet.GateNetV38 import GateNetV38
from modelzoo.backend.tensor.gatenet.GateNetV39 import GateNetV39
from modelzoo.backend.tensor.gatenet.GateNetV4 import GateNetV4
from modelzoo.backend.tensor.gatenet.GateNetV40 import GateNetV40
from modelzoo.backend.tensor.gatenet.GateNetV41 import GateNetV41
from modelzoo.backend.tensor.gatenet.GateNetV42 import GateNetV42
from modelzoo.backend.tensor.gatenet.GateNetV43 import GateNetV43
from modelzoo.backend.tensor.gatenet.GateNetV44 import GateNetV44
from modelzoo.backend.tensor.gatenet.GateNetV45 import GateNetV45
from modelzoo.backend.tensor.gatenet.GateNetV46 import GateNetV46
from modelzoo.backend.tensor.gatenet.GateNetV47 import GateNetV47
from modelzoo.backend.tensor.gatenet.GateNetV48 import GateNetV48
from modelzoo.backend.tensor.gatenet.GateNetV49 import GateNetV49
from modelzoo.backend.tensor.gatenet.GateNetV5 import GateNetV5
from modelzoo.backend.tensor.gatenet.GateNetV50 import GateNetV50
from modelzoo.backend.tensor.gatenet.GateNetV51 import GateNetV51
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
    architectures = {
        'GateNetV0': GateNetV0,
        'GateNetV1': GateNetV1,
        'GateNetV2': GateNetV2,
        'GateNetV3': GateNetV3,
        'GateNetV4': GateNetV4,
        'GateNetV5': GateNetV5,
        'GateNetV6': GateNetV6,
        'GateNetV7': GateNetV7,
        'GateNetV8': GateNetV8,
        'GateNetV9': GateNetV9,
        'GateNetV10': GateNetV10,
        'GateNetV11': GateNetV11,
        'GateNetV12': GateNetV12,
        'GateNetV13': GateNetV13,
        'GateNetV14': GateNetV14,
        'GateNetV15': GateNetV15,
        'GateNetV16': GateNetV16,
        'GateNetV17': GateNetV17,
        'GateNetV18': GateNetV18,
        'GateNetV19': GateNetV19,
        'GateNetV20': GateNetV20,
        'GateNetV21': GateNetV21,
        'GateNetV22': GateNetV22,
        'GateNetV23': GateNetV23,
        'GateNetV24': GateNetV24,
        'GateNetV25': GateNetV25,
        'GateNetV26': GateNetV26,
        'GateNetV28': GateNetV28,
        'GateNetV29': GateNetV29,
        'GateNetV30': GateNetV30,
        'GateNetV31': GateNetV31,
        'GateNetV32': GateNetV32,
        'GateNetV33': GateNetV33,
        'GateNetV34': GateNetV34,
        'GateNetV35': GateNetV35,
        'GateNetV36': GateNetV36,
        'GateNetV37': GateNetV37,
        'GateNetV38': GateNetV38,
        'GateNetV39': GateNetV39,
        'GateNetV40': GateNetV40,
        'GateNetV41': GateNetV41,
        'GateNetV42': GateNetV42,
        'GateNetV43': GateNetV43,
        'GateNetV44': GateNetV44,
        'GateNetV45': GateNetV45,
        'GateNetV46': GateNetV46,
        'GateNetV47': GateNetV47,
        'GateNetV48': GateNetV48,
        'GateNetV49': GateNetV49,
        'GateNetV50': GateNetV50,
        'GateNetV51': GateNetV51,
    }

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self.norm[0], self.norm[1], 3

    @staticmethod
    def create(model_name,
               norm=(416, 416),
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
        loss = GateDetectionLoss(
            grid=grid,
                                 n_boxes=n_boxes,
                                 n_polygon=4,
                                 weight_loc=scale_coor,
                                 weight_conf=scale_conf,
                                 weight_prob=scale_prob,
                                 weight_noobj=scale_noob)
        net = GateNet.architectures[model_name](loss=loss,
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
