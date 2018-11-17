import numpy as np

from etc.mulitply_adds import count_operations
from modelzoo.models.Postprocessor import Postprocessor
from modelzoo.models.Predictor import Predictor
from modelzoo.models.Preprocessor import Preprocessor
from modelzoo.models.gatenet import GateNet3x3V3
from modelzoo.models.gatenet import GateNetV10
from modelzoo.models.gatenet import GateNetV11
from modelzoo.models.gatenet import GateNetV14
from modelzoo.models.gatenet import GateNetV18
from modelzoo.models.gatenet import GateNetV19
from modelzoo.models.gatenet import GateNetV29, GateNetV30, GateNetV12, GateNetV13, GateNetV37, GateNetBase, GateNetV51, \
    GateNetV24, GateNetV39
from modelzoo.models.gatenet import GateNetV31
from modelzoo.models.gatenet import GateNetV40
from modelzoo.models.gatenet import GateNetV41
from modelzoo.models.gatenet import GateNetV43
from modelzoo.models.gatenet import GateNetV46
from modelzoo.models.gatenet import GateNetV48
from modelzoo.models.gatenet import GateNetV50
from modelzoo.models.gatenet import GateNetV9
from modelzoo.models.gatenet.GateDetectionLoss import GateDetectionLoss
from modelzoo.models.gatenet.GateNet3x3 import GateNet3x3
from modelzoo.models.gatenet.GateNet3x3V2 import GateNet3x3V2
from modelzoo.models.gatenet.GateNetBase import GateNetBase
from modelzoo.models.gatenet.GateNetDecoder import GateNetDecoder
from modelzoo.models.gatenet.GateNetEncoder import GateNetEncoder
from modelzoo.models.gatenet.GateNetFC import GateNetFC
from modelzoo.models.gatenet.GateNetSingle import GateNetSingle
from modelzoo.models.gatenet.GateNetV0 import GateNetV0
from modelzoo.models.gatenet.GateNetV1 import GateNetV1
from modelzoo.models.gatenet.GateNetV15 import GateNetV15
from modelzoo.models.gatenet.GateNetV16 import GateNetV16
from modelzoo.models.gatenet.GateNetV17 import GateNetV17
from modelzoo.models.gatenet.GateNetV2 import GateNetV2
from modelzoo.models.gatenet.GateNetV20 import GateNetV20
from modelzoo.models.gatenet.GateNetV21 import GateNetV21
from modelzoo.models.gatenet.GateNetV22 import GateNetV22
from modelzoo.models.gatenet.GateNetV23 import GateNetV23
from modelzoo.models.gatenet.GateNetV25 import GateNetV25
from modelzoo.models.gatenet.GateNetV26 import GateNetV26
from modelzoo.models.gatenet.GateNetV28 import GateNetV28
from modelzoo.models.gatenet.GateNetV3 import GateNetV3
from modelzoo.models.gatenet.GateNetV32 import GateNetV32
from modelzoo.models.gatenet.GateNetV33 import GateNetV33
from modelzoo.models.gatenet.GateNetV34 import GateNetV34
from modelzoo.models.gatenet.GateNetV35 import GateNetV35
from modelzoo.models.gatenet.GateNetV36 import GateNetV36
from modelzoo.models.gatenet.GateNetV38 import GateNetV38
from modelzoo.models.gatenet.GateNetV4 import GateNetV4
from modelzoo.models.gatenet.GateNetV42 import GateNetV42
from modelzoo.models.gatenet.GateNetV44 import GateNetV44
from modelzoo.models.gatenet.GateNetV45 import GateNetV45
from modelzoo.models.gatenet.GateNetV47 import GateNetV47
from modelzoo.models.gatenet.GateNetV49 import GateNetV49
from modelzoo.models.gatenet.GateNetV5 import GateNetV5
from modelzoo.models.gatenet.GateNetV6 import GateNetV6
from modelzoo.models.gatenet.GateNetV7 import GateNetV7
from modelzoo.models.gatenet.GateNetV8 import GateNetV8
from utils.imageprocessing.transform.ImgTransform import ImgTransform


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
        'GateNetSingle': GateNetSingle,
        'GateNet3x3': GateNet3x3,
        'GateNetFC': GateNetFC,
        'GateNet3x3V2': GateNet3x3V2,
        'GateNet3x3V3': GateNet3x3V3,
    }

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self.norm[0], self.norm[1], 3

    @staticmethod
    def create_by_arch(architecture,
                       norm=(416, 416),
                       input_channels=3,
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
                       n_polygon=4,
                       preprocessor=None,
                       resume_training=False):
        n_boxes = [len(a) for a in anchors]

        loss = GateDetectionLoss(
            n_polygon=n_polygon,
            weight_loc=scale_coor,
            weight_obj=scale_conf,
            weight_prob=scale_prob,
            weight_noobj=scale_noob)
        net = GateNetBase(architecture=architecture,
                          loss=loss,
                          anchors=anchors,
                          img_shape=norm,
                          weight_file=weight_file,
                          n_boxes=n_boxes,
                          n_polygon=n_polygon,
                          input_channels=input_channels,
                          resume_training=resume_training
                          )

        return GateNet(net,
                       anchors=anchors,
                       batch_size=batch_size,
                       grid=net.grid,
                       norm=norm,
                       conf_thresh=conf_thresh,
                       color_format=color_format,
                       augmenter=augmenter,
                       n_polygon=n_polygon,
                       preprocessing=preprocessor,
                       architecture=architecture)

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
            anchors = np.array([[[1.08, 1.19],
                                 [3.42, 4.41],
                                 [6.63, 11.38],
                                 [9.42, 5.11],
                                 [16.62, 10.52]]])

        n_boxes = int(np.ceil(anchors.size / 2))
        loss = GateDetectionLoss(
            n_polygon=n_polygon,
            weight_loc=scale_coor,
            weight_obj=scale_conf,
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
                 n_polygon=4,
                 preprocessing=None,
                 architecture=None,
                 ):
        self.architecture = architecture
        self.color_format = color_format

        self.anchors = anchors
        self.batch_size = batch_size
        self.grid = grid
        self.norm = norm
        self.conf_thresh = conf_thresh
        self.n_boxes = [int(np.ceil(len(a) / 2)) for a in anchors]
        self._output_shape = grid[0][0] * grid[0][1], self.n_boxes * (n_polygon + 1)

        encoder = GateNetEncoder(img_norm=norm,
                                 anchor_dims=anchors,
                                 grids=grid,
                                 n_polygon=n_polygon)
        preprocessor = Preprocessor(augmentation=augmenter,
                                    encoder=encoder,
                                    img_shape=self.norm,
                                    n_classes=1,
                                    color_format=color_format,
                                    preprocessing=preprocessing)

        decoder = GateNetDecoder(norm=norm,
                                 grid=grid,
                                 anchor_dims=anchors,
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

    @property
    def n_multiply_adds(self):
        return count_operations(self.architecture, self.input_shape)
