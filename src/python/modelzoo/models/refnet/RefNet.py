import numpy as np

from modelzoo.backend.tensor.refnet.RefNetBase import RefNetBase
from modelzoo.backend.tensor.gatenet.GateDetectionLoss import GateDetectionLoss
from modelzoo.models.Postprocessor import Postprocessor
from modelzoo.models.Predictor import Predictor
from modelzoo.models.Preprocessor import Preprocessor
from modelzoo.models.gatenet.GateNetDecoder import GateNetDecoder
from modelzoo.models.refnet.RefNetDecoder import RefNetDecoder
from modelzoo.models.refnet.RefNetEncoder import RefNetEncoder
from modelzoo.models.refnet.RefNetPreprocessor import RefNetPreprocessor
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ObjectLabel import ObjectLabel


class RefNet(Predictor):

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self.norm[0], self.norm[1], 3

    @staticmethod
    def create_by_arch(architecture,
                       norm=(416, 416),
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
                       n_rois=5,
                       crop_size=(52, 52)):
        n_boxes = anchors.shape[1]
        loss = GateDetectionLoss(
            n_boxes=n_boxes,
            n_polygon=n_polygon,
            weight_loc=scale_coor,
            weight_conf=scale_conf,
            weight_prob=scale_prob,
            weight_noobj=scale_noob)
        net = RefNetBase(architecture=architecture,
                         crop_size=crop_size[0],
                         n_rois=n_rois,
                         loss=loss,
                         anchors=anchors,
                         norm=norm,
                         weight_file=weight_file,
                         n_boxes=n_boxes,
                         n_polygon=n_polygon)

        return RefNet(net,
                      anchors=anchors,
                      batch_size=batch_size,
                      grid=net.grid,
                      norm=norm,
                      conf_thresh=conf_thresh,
                      color_format=color_format,
                      augmenter=augmenter,
                      n_polygon=n_polygon,
                      crop_size=crop_size,
                      n_roi=n_rois)

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
                 crop_size=(52, 52),
                 n_roi=5):
        self.color_format = color_format
        if anchors is None:
            anchors = np.array([[[1.3221, 1.73145],
                                 [3.19275, 4.00944],
                                 [5.05587, 8.09892],
                                 [9.47112, 4.84053],
                                 [11.2364, 10.0071]]])

        ObjectLabel.classes = ['gate']

        self.anchors = anchors
        self.batch_size = batch_size
        self.grid = grid
        self.norm = norm
        self.conf_thresh = conf_thresh
        self.n_boxes = anchors.shape[0]
        self._output_shape = grid[0][0] * grid[0][1], self.n_boxes * (n_polygon + 1)

        encoder = RefNetEncoder(img_norm=norm,
                                anchor_dims=anchors,
                                n_regions=n_roi,
                                grids=grid,
                                n_boxes=self.n_boxes,
                                n_polygon=n_polygon,
                                crop_size=crop_size)
        preprocessor = RefNetPreprocessor(augmenter=augmenter,
                                          encoder=encoder,
                                          img_shape=self.norm,
                                          n_classes=1,
                                          color_format=color_format)

        decoder = RefNetDecoder(norm=norm,
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
