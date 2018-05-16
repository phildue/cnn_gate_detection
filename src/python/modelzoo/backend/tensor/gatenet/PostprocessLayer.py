from keras.engine import Layer
import keras.backend as K


class PostprocessLayer(Layer):

    def __init__(self, grid=(13, 13), n_boxes=5, norm=(416, 416), n_polygon=4,
                 iou_thresh=0.4, n_boxes_nms_max=13 * 13 * 5, **kwargs):
        self.n_boxes_max = n_boxes_nms_max
        self.n_boxes = n_boxes
        self.grid = grid

        self.iou_thresh = iou_thresh
        self.n_polygon = n_polygon
        self.norm = norm
        super(PostprocessLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(PostprocessLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return self._postprocess_pred(x)

    def compute_output_shape(self, input_shape):
        return 1, self.grid[0] * self.grid[1] * self.n_boxes, self.n_polygon + 1

    def _decode_coord(self, coord_t):
        offset_y, offset_x = K.np.mgrid[:self.grid[0], :self.grid[1]]
        offset_y = K.constant(offset_y, K.tf.float32)
        offset_x = K.constant(offset_x, K.tf.float32)
        offset_x = K.expand_dims(offset_x, -1)
        offset_x = K.expand_dims(offset_x, 0)
        offset_x = K.tile(offset_x, (1, 1, 1, self.n_boxes))

        offset_y = K.expand_dims(offset_y, -1)
        offset_y = K.expand_dims(offset_y, 0)
        offset_y = K.tile(offset_y, (1, 1, 1, self.n_boxes))

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
        y_pred = K.reshape(y_pred, [1, self.grid[0], self.grid[1], self.n_boxes, self.n_polygon + 1])
        coord_pred_t = y_pred[:, :, :, :, :4]
        conf_pred_t = y_pred[:, :, :, :, 4]
        coord_pred_dec_t = self._decode_coord(coord_pred_t)

        coord_pred_reshape_t = K.reshape(coord_pred_dec_t, (-1, 4))
        conf_pred_reshape_t = K.reshape(conf_pred_t, (-1, 1))

        idx = K.tf.image.non_max_suppression(K.cast(coord_pred_reshape_t, K.tf.float32),
                                             K.cast(K.flatten(conf_pred_reshape_t), K.tf.float32),
                                             self.n_boxes_max, self.iou_thresh,
                                             'NonMaxSuppression')
        idx = K.expand_dims(idx, 1)
        class_pred_nms = K.tf.gather_nd(conf_pred_reshape_t, idx)
        class_pred_nms = K.tf.scatter_nd(idx, class_pred_nms, shape=K.shape(conf_pred_reshape_t))
        class_pred_nms = K.expand_dims(class_pred_nms, 0)
        coord_pred_reshape_t = K.expand_dims(coord_pred_reshape_t, 0)

        out = K.concatenate([coord_pred_reshape_t, class_pred_nms])
        out = K.reshape(out, (1, -1, self.n_polygon + 1))

        return out
