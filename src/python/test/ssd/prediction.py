import keras.backend as K

from modelzoo.models.ssd.SSD import SSD
from utils.fileaccess.VocGenerator import VocGenerator
from utils.imageprocessing.Backend import resize, annotate_text
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED
from utils.labels.ObjectLabel import ObjectLabel
from utils.workdir import work_dir

work_dir()

batch_size = 5
ssd = SSD.ssd_test(n_classes=20, conf_thresh=0, batch_size=batch_size,
                   weight_file='resource/models/vgg16_imagenet.h5')
dataset = VocGenerator(batch_size=batch_size).generate()
class_names = ['Background'] + ObjectLabel.classes.copy()
while True:
    batch = next(dataset)

    labels_true_t = []
    imgs_enc = []
    for i in range(batch_size):
        label_true = batch[i][1]
        img = batch[i][0]
        img, label_true = resize(img, (300, 300), label=label_true)
        label_true_t = ssd.preprocessor.encoder.encode_label(label_true)
        label_true_t = K.np.expand_dims(label_true_t, 0)
        labels_true_t.append(label_true_t)
        img_enc = ssd.preprocessor.encoder.encode_img(img)
        imgs_enc.append(img_enc)

    label_true_t = K.np.concatenate(labels_true_t)
    img_enc = K.np.concatenate(imgs_enc)

    label_pred_t = ssd.net.predict(img_enc)
    label_pred_t = label_true_t.copy()
    label_pred_t[:, :, :-17] *= 10
    label_pred_t[label_pred_t[:, :, 0] == 0, :-13] = K.np.array(
        [1, 1.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    label_pred_t[:, -16] += 3.0
    sess = K.tf.InteractiveSession()
    localization_loss_t = ssd.loss.localization_loss(y_pred=K.constant(label_pred_t),
                                                     y_true=K.constant(label_true_t))
    conf_loss_pos_t = ssd.loss.conf_loss_positives(y_pred=K.constant(label_pred_t),
                                                   y_true=K.constant(label_true_t))
    conf_loss_neg_t = ssd.loss.conf_loss_negatives(y_pred=K.constant(label_pred_t),
                                                   y_true=K.constant(label_true_t))
    loss_t = ssd.loss.compute(y_pred=K.constant(label_pred_t), y_true=K.constant(label_true_t))
    conf_loss_pos = K.get_session().run(conf_loss_pos_t)
    conf_loss_neg = K.get_session().run(conf_loss_neg_t)
    localization_loss = K.get_session().run(localization_loss_t)
    print("Loss:", loss_t.eval())

    for i in range(batch_size):
        img, label_true, _ = batch[i]
        img_res, label_true = resize(img, (600, 600), label=label_true)
        # show_anchors(ssd.preprocessor.anchors_t, label_true_t[i], img_res)
        label_pred = ssd.decoder.decode_netout_to_label(label_pred_t[i])
        _, label_pred = resize(img, scale_x=2.0, scale_y=2.0, label=label_pred)
        img_res = annotate_text("LocLoss: {:.4f}"
                                "PosClassLoss: {:.4f}"
                                "NegClassLoss: {:.4f}".format(localization_loss[i], conf_loss_pos[i], conf_loss_neg[i])
                                , img_res, thickness=2, color=(0, 0, 0), xy=(10, 10))
        show(img_res, labels=[label_true, label_pred], colors=[COLOR_GREEN, COLOR_RED])
