import numpy as np


# noinspection PyDefaultArgument
from modelzoo.augmentation.SSDAugmenter import SSDAugmenter
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
               conf_thresh=0.7,
               iou_thresh_nms=0.45,
               top_k=400,
               batch_size=5,
               color_format='bgr',
               alpha=1.0,
               neg_pos_ratio=3,
               neg_min=0,
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
        n_classes = n_classes + 1

        loss = MultiboxLoss(batch_size=batch_size,
                            n_negatives_min=neg_min,
                            negative_positive_ratio=neg_pos_ratio,
                            loc_class_error_weight=alpha)
        net = SSD300(loss=loss,
                     image_size=image_shape,
                     weight_file=weight_file,
                     n_classes=n_classes,
                     n_boxes=n_boxes)

        return SSD(img_shape=image_shape,
                   n_classes=n_classes,
                   aspect_ratios=aspect_ratios,
                   scales=scales,
                   clip_boxes=clip_boxes,
                   variances=variances,
                   iou_thresh_match=iou_thresh_match,
                   iou_thresh_nms=iou_thresh_nms,
                   confidence_thresh=conf_thresh,
                   color_format=color_format,
                   top_k=top_k,
                   net=net)

    @staticmethod
    def ssd7(image_shape=(300, 300, 3), weight_file=None, n_classes=20, clip_boxes=False,
             variances=[1.0, 1.0, 1.0, 1.0],
             iou_thresh_match=0.5,
             conf_thresh=0.7,
             iou_thresh_nms=0.45,
             top_k=400, batch_size=5, color_format='bgr', alpha=1.0, neg_pos_ratio=3, neg_min=0):

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

        net = SSD7(loss=loss,
                   image_size=image_shape,
                   weight_file=weight_file,
                   n_classes=n_classes,
                   n_boxes=n_boxes,
                   )

        return SSD(img_shape=image_shape,
                   n_classes=n_classes,
                   aspect_ratios=aspect_ratios,
                   scales=scales,
                   clip_boxes=clip_boxes,
                   variances=variances,
                   iou_thresh_match=iou_thresh_match,
                   iou_thresh_nms=iou_thresh_nms,
                   confidence_thresh=conf_thresh,
                   color_format=color_format,
                   top_k=top_k,
                   net=net)

    @staticmethod
    def ssd_test(image_shape=(300, 300, 3), weight_file=None, n_classes=20, clip_boxes=False,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 iou_thresh_match=0.5,
                 conf_thresh=0.7,
                 iou_thresh_nms=0.45,
                 top_k=400, batch_size=5, color_format='bgr', alpha=1.0, neg_pos_ratio=3, neg_min=0):

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
        net = SSDTestNet(loss=loss,
                         image_size=image_shape,
                         weight_file=weight_file,
                         n_classes=n_classes,
                         n_boxes=n_boxes)

        return SSD(img_shape=image_shape,
                   n_classes=n_classes,
                   aspect_ratios=aspect_ratios,
                   scales=scales,
                   clip_boxes=clip_boxes,
                   variances=variances,
                   iou_thresh_match=iou_thresh_match,
                   iou_thresh_nms=iou_thresh_nms,
                   confidence_thresh=conf_thresh,
                   color_format=color_format,
                   top_k=top_k,
                   net=net)

    def __init__(self, img_shape, n_classes,
                 net: SSDNet,
                 scales,
                 aspect_ratios,
                 clip_boxes=False,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 iou_thresh_match=0.5,
                 confidence_thresh=0.7,
                 iou_thresh_nms=0.45,
                 color_format='bgr',
                 top_k=400):

        self.variances = variances
        self.clip_boxes = clip_boxes
        self.img_shape = img_shape
        self.n_classes = n_classes

        self.predictor_sizes = net.predictor_sizes

        self.aspect_ratios = aspect_ratios

        self.scales = scales

        anchors_t = self.generate_anchors_t()
        encoder = SSDEncoder(img_shape,
                             self.n_classes,
                             anchors_t,
                             variances,
                             iou_thresh_match)
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

    def generate_anchor_t(self,
                          feature_map_size,
                          aspect_ratios,
                          scale, next_scale):

        """
        Compute an array of the spatial positions and sizes of the anchor boxes for one particular classification
        layer of size `feature_map_size == [feature_map_height, feature_map_width]`.

        :param feature_map_size:  tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
        :param aspect_ratios: A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
        :param scale: A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
        :return: tensor(feature_map_h*feature_map_w*#boxes,4)
        """

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        aspect_ratios = np.sort(aspect_ratios)
        size = min(self.img_shape[:1])
        n_boxes = len(aspect_ratios)
        if 1.0 in aspect_ratios:
            n_boxes += 1

        # Compute the grid of box center points. They are identical for all aspect ratios
        cell_height = self.img_shape[0] / feature_map_size[0]
        cell_width = self.img_shape[1] / feature_map_size[1]
        cx = np.linspace(cell_width / 2, self.img_shape[1] - cell_width / 2, feature_map_size[1])
        cy = np.linspace(cell_height / 2, self.img_shape[0] - cell_height / 2, feature_map_size[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy

        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for i, ar in enumerate(aspect_ratios):
            if ar == 1.0:
                scale_sqrt = np.sqrt(scale * next_scale)
                w = scale_sqrt * size * np.sqrt(ar)
                h = scale_sqrt * size / np.sqrt(ar)
                wh_list.append((w, h))

            w = scale * size * np.sqrt(ar)
            h = scale * size / np.sqrt(ar)
            wh_list.append((w, h))

        wh_list = np.array(wh_list)

        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 1]]
            x_coords[x_coords >= self.img_shape[1]] = self.img_shape[1] - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 1]] = x_coords
            y_coords = boxes_tensor[:, :, :, [2, 3]]
            y_coords[y_coords >= self.img_shape[0]] = self.img_shape[0] - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [2, 3]] = y_coords

        boxes_tensor = np.reshape(boxes_tensor, (-1, 4))

        return boxes_tensor

    def generate_anchors_t(self):
        """
        Generates a tensor that contains the anchor box coordinates.
        The shape and content is determined by the number of predictor layers used in the
        model architecture and the amount of anchor boxes for each predictor.

        Returns:
            A Numpy array of shape `(#boxes, 4)` [cx,cy,w,h]
        """

        boxes_tensor = []
        for i in range(len(self.predictor_sizes)):
            boxes = self.generate_anchor_t(feature_map_size=self.predictor_sizes[i],
                                           aspect_ratios=self.aspect_ratios[i],
                                           scale=self.scales[i],
                                           next_scale=self.scales[i + 1])
            boxes_tensor.append(boxes)

        return np.concatenate(boxes_tensor, axis=0)
