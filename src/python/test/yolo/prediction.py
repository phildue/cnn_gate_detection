import keras.backend as K

from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize, annotate_text
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED
from utils.workdir import work_dir

work_dir()

batch_size = 4
yolo = Yolo.yolo_v2(class_names=['gate'], conf_thresh=0, batch_size=batch_size,
                    weight_file='logs/yolov2_25k/YoloV2.h5')
dataset = GateGenerator(directories='resource/samples/mult_gate_aligned/', batch_size=batch_size, color_format='yuv',
                        shuffle=True, n_samples=1000).generate()

while True:
    batch = next(dataset)

    labels_true_t = []
    imgs_enc = []
    for i in range(batch_size):
        label_true = batch[i][1]
        img = batch[i][0]
        img, label_true = resize(img, (416, 416), label=label_true)
        label_true_t = yolo.preprocessor.encoder.encode_label(label_true)
        label_true_t = K.np.expand_dims(label_true_t, 0)
        labels_true_t.append(label_true_t)
        img_enc = yolo.preprocessor.encoder.encode_img(img)
        imgs_enc.append(img_enc)

    label_true_t = K.np.concatenate(labels_true_t)
    img_enc = K.np.concatenate(imgs_enc)

    label_pred_t = yolo.net.predict(img_enc)
    # label_pred_t = label_true_t.copy()
    # label_pred_t[label_pred_t[:, :, 0] == 0, :-4] = K.np.array(
    #     [0.2, 0.6, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # # label_pred_t[:, -3] += 0.2
    sess = K.tf.InteractiveSession()
    loss = yolo.net.loss.compute(y_pred=K.constant(label_pred_t), y_true=K.constant(label_true_t)).eval()
    loc_loss = yolo.net.loss.localization_loss(y_pred=K.constant(label_pred_t),
                                               y_true=K.constant(label_true_t)).eval()
    conf_loss = yolo.net.loss.confidence_loss(y_pred=K.constant(label_pred_t), y_true=K.constant(label_true_t)).eval()
    class_loss = yolo.net.loss.class_loss(y_pred=K.constant(label_pred_t), y_true=K.constant(label_true_t)).eval()
    print("Loss:", loss)

    for i in range(batch_size):
        img, label_true = batch[i]
        img_res, label_true = resize(img, (600, 600), label=label_true)
        # show_anchors(ssd.preprocessor.anchors_t, label_true_t[i], img_res)
        label_pred = yolo.postprocessor.decoder.decode_netout_to_label(label_pred_t[i])
        _, label_pred = resize(img, scale_x=2.0, scale_y=2.0, label=label_pred)
        img_res = annotate_text("Loc Loss: {:.4f} Conf Loss {:.4} Class Loss: {:.4f}".format(loc_loss[i],
                                                                                             conf_loss[i],
                                                                                             class_loss[i])
                                , img_res, thickness=2, color=(0, 0, 0), xy=(10, 10))
        show(img_res, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_RED])
