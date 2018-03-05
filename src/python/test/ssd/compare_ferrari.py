import keras.backend as K
from frontend.models.ssd.SSD import SSD
from workdir import work_dir

from src.python.test.ssd import SSDBoxEncoder
from src.python.test.ssd import SSDLoss
from src.python.utils.fileaccess import VocGenerator

work_dir()
batch_size = 4
dataset = VocGenerator(batch_size=batch_size, shuffle=False, start_idx=25).generate()
batch = next(dataset)

ssd = SSD.ssd300(batch_size=2)

batch_enc_my = ssd.encoder.encode_label_batch([b[1] for b in batch])

# reformat batch how ferrari expects
img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
n_classes = 21  # Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
          1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets, the factors for the MS COCO dataset are smaller, namely [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
aspect_ratios = [[0.5, 1.0, 2.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0]]  # The anchor box aspect ratios used in the original SSD300
two_boxes_for_ar1 = True
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = True
predictor_sizes = [[37, 37],
                   [18, 18],
                   [9, 9],
                   [5, 5],
                   [3, 3],
                   [1, 1]]
ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)
#
# ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
#                 for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
#                 to the respective image, and the data for each ground truth bounding box has the format
#                 `(class_id, xmin, xmax, ymin, ymax)`, and `class_id` must be an integer greater than 0 for all boxes
#                 as class_id 0 is reserved for the background class.

ground_truth_labels = []
for i in range(batch_size):
    label = batch[i][1]
    img_label_ferrari = []
    for obj in label.objects:
        obj_label = K.np.array([obj.class_id + 1, obj.x_min, obj.x_max, obj.y_min, obj.y_max])
        img_label_ferrari.append(obj_label)
    ground_truth_labels.append(K.np.vstack(img_label_ferrari))

batch_enc_ferrari = ssd_box_encoder.encode_y(ground_truth_labels)

diff = batch_enc_ferrari[:, :, :25] - batch_enc_my

fg_ferrari = batch_enc_ferrari[0, batch_enc_ferrari[0, :, 0] == 0]
fg_my = batch_enc_my[0, batch_enc_my[0, :, 0] == 0]

obj_ferrari = batch_enc_ferrari[K.np.max(batch_enc_ferrari[:, :, 1:21], -1) > 0]
obj_my = batch_enc_my[K.np.max(batch_enc_my[:, :, 1:21], -1) > 0]

loss_my = ssd.loss.compute(K.constant(batch_enc_my[:2]), K.constant(batch_enc_my[2:4]))
loc_loss_my = ssd.loss.localization_loss(K.constant(batch_enc_my[:2]), K.constant(batch_enc_my[2:4]))
loss_ferrari = SSDLoss().compute_loss(K.constant(batch_enc_ferrari[:2]), K.constant(batch_enc_ferrari[2:4]))

sess = K.tf.InteractiveSession()
loss_my_res = loss_my.eval()
loss_my_ferrari = loss_ferrari.eval()
loc_loss_my = loc_loss_my.eval()
sess.close()
print("My loss", loss_my_res)
print("Ferrari Loss", loss_my_ferrari)
print(loc_loss_my)


def log_loss(y_true, y_pred):
    y_pred = K.np.maximum(y_pred, 1e-15)
    # Compute the log loss
    log_loss = -K.np.sum(y_true * K.np.log(y_pred), axis=-1)
    return log_loss


my_loc_loss = log_loss(batch_enc_my[:2, :, -4:], batch_enc_my[2:, :, -4:])
ferrari_loc_loss = log_loss(batch_enc_ferrari[:2, :, -12:-8], batch_enc_ferrari[2:, :, -12:-8])
