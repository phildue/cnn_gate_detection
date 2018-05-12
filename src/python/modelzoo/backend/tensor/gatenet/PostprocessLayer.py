from keras.engine import Layer
import keras.backend as K


class PostprocessLayer(Layer):

    def __init__(self, grid=(13, 13), batch_size=8, n_boxes=5, norm=(416, 416), n_classes=20,
                 iou_thresh=0.4,
                 max_boxes_nms=None, **kwargs):

        if max_boxes_nms is None:
            max_boxes_nms = self.n_boxes * self.grid[0] * self.grid[1]

        self.max_boxes_nms = max_boxes_nms
        self.iou_thresh = iou_thresh
        self.n_classes = n_classes
        self.norm = norm
        self.n_boxes = n_boxes
        self.grid = grid
        super(PostprocessLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(PostprocessLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return self._postprocess_pred(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.max_boxes_nms, self.n_classes + 4

    def _decode_coord(self, coord_t):
        batch_size = K.shape(coord_t)[0]
        offset_y, offset_x = K.np.mgrid[:self.grid[0], :self.grid[1]]
        offset_y = K.constant(offset_y, K.tf.float32)
        offset_x = K.constant(offset_x, K.tf.float32)
        offset_x = K.expand_dims(offset_x, -1)
        offset_x = K.expand_dims(offset_x, 0)
        offset_x = K.tile(offset_x, (batch_size, 1, 1, self.n_boxes))

        offset_y = K.expand_dims(offset_y, -1)
        offset_y = K.expand_dims(offset_y, 0)
        offset_y = K.tile(offset_y, (batch_size, 1, 1, self.n_boxes))

        coord_t_cx = coord_t[:, :, :, :, 0] + offset_x
        coord_t_cy = coord_t[:, :, :, :, 1] + offset_y
        coord_t_cx = coord_t_cx * (self.norm[1] / self.grid[1])
        coord_t_w = coord_t[:, :, :, :, 2] * (self.norm[1] / self.grid[1])
        coord_t_cy = coord_t_cy * (self.norm[0] / self.grid[0])
        coord_t_h = coord_t[:, :, :, :, 3] * (self.norm[0] / self.grid[0])

        coord_t_cy = self.norm[0] - coord_t_cy

        coord_t_xmin = coord_t_cx - coord_t_w / 2
        coord_t_ymin = coord_t_cy - coord_t_h / 2
        coord_t_xmax = coord_t_cx + coord_t_w / 2
        coord_t_ymax = coord_t_cy + coord_t_h / 2

        coord_t_xmin = K.expand_dims(coord_t_xmin, -1)
        coord_t_ymin = K.expand_dims(coord_t_ymin, -1)
        coord_t_xmax = K.expand_dims(coord_t_xmax, -1)
        coord_t_ymax = K.expand_dims(coord_t_ymax, -1)
        coord_dec_t = K.concatenate([coord_t_xmin, coord_t_ymin, coord_t_xmax, coord_t_ymax], -1)

        return coord_dec_t

    def _postprocess_pred(self, y_pred):
        coord_pred_t = y_pred[:, :, :, :, :4]
        conf_pred_t = y_pred[:, :, :, :, 4]
        batch_size = K.shape(y_pred)[0]
        coord_pred_dec_t = self._decode_coord(coord_pred_t)

        coord_pred_reshape_t = K.reshape(coord_pred_dec_t, (batch_size, -1, 4))
        conf_pred_reshape_t = K.reshape(conf_pred_t, (batch_size, -1, 1))

        class_pred_nms_batch = self.non_max_suppression_batch(coord_pred_t,
                                                              conf_pred_reshape_t,
                                                              batch_size,
                                                              self.max_boxes_nms,
                                                              self.iou_thresh)

        return coord_pred_reshape_t, class_pred_nms_batch

    @staticmethod
    def non_max_suppression_batch(coord_pred_t, class_pred_t, batch_size, n_boxes_max, iou_thresh):
        """
        Performs non-max suppression on a batch of labels.
        :param coord_pred_t: tensor(#batch,#boxes,4) containing box coordinates in min-max format
        :param class_pred_t: tensor(#batch,#boxes,#classes) containing predictions per box
        :param batch_size: size of the batch
        :param n_boxes_max: max number of boxes to keep
        :param iou_thresh: intersection-over-union threshold when a box counts as overlapping
        :return: tensor(#batch,#boxes,#classes) where the suppressed boxes are all 0s
        """
        conf_pred_t = K.max(class_pred_t, axis=-1)

        class_pred_nms_batch = []
        for i in range(batch_size):
            idx = K.tf.image.non_max_suppression(K.cast(coord_pred_t[i], K.tf.float32),
                                                 K.cast(K.flatten(conf_pred_t[i]), K.tf.float32),
                                                 n_boxes_max, iou_thresh,
                                                 'NonMaxSuppression')
            idx = K.expand_dims(idx, 1)
            class_pred_nms = K.tf.gather_nd(class_pred_t[i], idx)
            class_pred_nms = K.tf.scatter_nd(idx, class_pred_nms, shape=K.shape(class_pred_t[0]))
            class_pred_nms = K.expand_dims(class_pred_nms, 0)
            class_pred_nms_batch.append(class_pred_nms)

        class_pred_nms_batch = K.concatenate(class_pred_nms_batch, 0)
        return class_pred_nms_batch
