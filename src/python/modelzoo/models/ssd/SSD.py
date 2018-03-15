import numpy as np

from modelzoo.backend.tensor.ssd.MultiboxLoss import MultiboxLoss
from modelzoo.backend.tensor.ssd.SSD300 import SSD300
from modelzoo.backend.tensor.ssd.SSD7 import SSD7
from modelzoo.backend.tensor.ssd.SSDNet import SSDNet
from modelzoo.backend.tensor.ssd.SSDTestNet import SSDTestNet
from modelzoo.models.Postprocessor import Postprocessor
from modelzoo.models.Predictor import Predictor
from modelzoo.models.Preprocessor import Preprocessor
from modelzoo.models.ssd.SSDDecoder import SSDDecoder
from modelzoo.models.ssd.SSDEncoder import SSDEncoder
# noinspection PyDefaultArgument
from utils.imageprocessing.transform.SSDAugmenter import SSDAugmenter


class SSD(Predictor):
    @property
    def input_shape(self):
        return self.img_shape

    @staticmethod
    def ssd300(image_shape=(300, 300, 3),
               weight_file=None,
               n_classes=20,
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               iou_thresh_match=0.5,
               iou_thresh_background=0.2,
               conf_thresh=0.7,
               iou_thresh_nms=0.45,
               batch_size=5,
               color_format='bgr',
               alpha=1.0,
               neg_pos_ratio=3,
               neg_min=0.2,
               scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                       1.05],
               aspect_ratios=[[0.5, 1.0, 2.0],
                              [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                              [0.5, 1.0, 2.0],
                              [0.5, 1.0, 2.0]],
               n_boxes={'conv4': 4,
                        'fc7': 6,
                        'conv8': 6,
                        'conv9': 6,
                        'conv10': 4,
                        'conv11': 4}):
        # We add the background class
        n_classes += 1

        loss = MultiboxLoss(batch_size=batch_size,
                            n_negatives_min=neg_min,
                            negative_positive_ratio=neg_pos_ratio,
                            loc_class_error_weight=alpha)
        net = SSD300(img_shape=image_shape,
                     variances=variances,
                     scales=scales,
                     aspect_ratios=aspect_ratios,
                     loss=loss,
                     weight_file=weight_file,
                     n_classes=n_classes,
                     n_boxes=n_boxes)

        return SSD(img_shape=image_shape,
                   n_classes=n_classes,
                   clip_boxes=clip_boxes,
                   variances=variances,
                   iou_thresh_match=iou_thresh_match,
                   iou_thresh_nms=iou_thresh_nms,
                   confidence_thresh=conf_thresh,
                   color_format=color_format,
                   net=net,
                   iou_thresh_background=iou_thresh_background)

    @staticmethod
    def ssd7(image_shape=(300, 300, 3), weight_file=None, n_classes=20, clip_boxes=False,
             variances=[1.0, 1.0, 1.0, 1.0],
             iou_thresh_match=0.5,
             iou_thresh_background=0.2,
             conf_thresh=0.7,
             iou_thresh_nms=0.45,
             batch_size=5, color_format='bgr', alpha=1.0, neg_pos_ratio=3, neg_min=0):
        aspect_ratios = [[1.0, 2.0, 3.0, 0.5, 0.33]] * 4

        n_boxes = {'conv4': 6,
                   'conv5': 6,
                   'conv6': 6,
                   'conv7': 6}

        scales = np.linspace(0.1, 0.9, len(n_boxes) + 1)

        # We add the background class
        n_classes = n_classes + 1

        loss = MultiboxLoss(batch_size=batch_size,
                            n_negatives_min=neg_min,
                            negative_positive_ratio=neg_pos_ratio,
                            loc_class_error_weight=alpha)

        net = SSD7(img_shape=image_shape,
                   variances=variances,
                   scales=scales,
                   aspect_ratios=aspect_ratios,
                   loss=loss,
                   weight_file=weight_file,
                   n_classes=n_classes,
                   n_boxes=n_boxes,
                   )

        return SSD(img_shape=image_shape,
                   n_classes=n_classes,
                   clip_boxes=clip_boxes,
                   variances=variances,
                   iou_thresh_match=iou_thresh_match,
                   iou_thresh_nms=iou_thresh_nms,
                   confidence_thresh=conf_thresh,
                   color_format=color_format,
                   net=net,
                   iou_thresh_background=iou_thresh_background)

    @staticmethod
    def ssd_test(image_shape=(300, 300, 3), weight_file=None, n_classes=20, clip_boxes=False,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 iou_thresh_match=0.5,
                 iou_thresh_background=0.2,
                 conf_thresh=0.7,
                 iou_thresh_nms=0.45,
                 batch_size=5, color_format='bgr', alpha=1.0, neg_pos_ratio=3, neg_min=0):
        aspect_ratios = [[1.1]] * 2

        n_boxes = {'conv6': 1,
                   'conv7': 1}

        scales = np.linspace(0.1, 0.9, len(n_boxes) + 1)

        # We add the background class
        n_classes = n_classes + 1

        loss = MultiboxLoss(batch_size=batch_size,
                            n_negatives_min=neg_min,
                            negative_positive_ratio=neg_pos_ratio,
                            loc_class_error_weight=alpha)
        net = SSDTestNet(img_shape=image_shape,
                         variances=variances,
                         scales=scales,
                         aspect_ratios=aspect_ratios,
                         loss=loss,
                         weight_file=weight_file,
                         n_classes=n_classes,
                         n_boxes=n_boxes)

        return SSD(img_shape=image_shape,
                   n_classes=n_classes,
                   clip_boxes=clip_boxes,
                   variances=variances,
                   iou_thresh_match=iou_thresh_match,
                   iou_thresh_nms=iou_thresh_nms,
                   confidence_thresh=conf_thresh,
                   color_format=color_format,
                   net=net,
                   iou_thresh_background=iou_thresh_background)

    def __init__(self, img_shape, n_classes,
                 net: SSDNet,
                 clip_boxes=False,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 iou_thresh_match=0.5,
                 confidence_thresh=0.7,
                 iou_thresh_nms=0.45,
                 color_format='bgr',
                 iou_thresh_background=0.2):
        self.variances = variances
        self.clip_boxes = clip_boxes
        self.img_shape = img_shape
        self.n_classes = n_classes

        self.predictor_sizes = net.anchors

        encoder = SSDEncoder(img_shape=img_shape,
                             n_classes=self.n_classes,
                             anchors_t=net.anchors,
                             variances=variances,
                             iou_thresh_match=iou_thresh_match,
                             iou_thresh_background=iou_thresh_background)

        preprocessor = Preprocessor(augmenter=SSDAugmenter(),
                                    encoder=encoder,
                                    img_shape=img_shape,
                                    n_classes=self.n_classes,
                                    color_format=color_format)

        decoder = SSDDecoder(img_shape)
        postprocessor = Postprocessor(decoder=decoder,
                                      conf_thresh=confidence_thresh,
                                      iou_thresh=iou_thresh_nms)

        super().__init__(preprocessor=preprocessor,
                         postprocessor=postprocessor,
                         net=net,
                         loss=net.loss,
                         encoder=encoder,
                         decoder=decoder)
