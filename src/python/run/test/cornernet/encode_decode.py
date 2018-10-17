import keras.backend as K

from modelzoo.models.cornernet import CornerNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.imageprocessing.Backend import resize, crop
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED, LEGEND_CORNERS, COLOR_BLUE
from utils.workdir import cd_work

cd_work()
batch_size = 4
dataset = GateGenerator(["resource/ext/samples/cyberzoo/"], batch_size=batch_size,
                        color_format='bgr', label_format='xml', n_samples=batch_size).generate()
batch = next(dataset)

predictor = CornerNet((64, 64), 4)

for i in range(batch_size):
    img = batch[i][0]
    label_true = batch[i][1]

    img_crop, label_crop = crop(img, (label_true.objects[0].x_min-10, label_true.objects[0].y_min-10),
                                (label_true.objects[0].x_max+10, label_true.objects[0].y_max+10), label=label_true)

    img_crop, label_crop = resize(img_crop, predictor.input_shape, label=label_crop)

    label_enc = predictor.encoder.encode_label(label_crop)
    label_enc_fake = label_enc.copy()
    label_enc_fake[0] -= 0.1
    label_dec = predictor.postprocessor.decoder.decode_netout_to_label(label_enc)
    label_dec_fake = predictor.postprocessor.decoder.decode_netout_to_label(label_enc_fake)

    loss = K.get_session().run(predictor.loss.compute(y_pred=K.constant(label_enc, K.tf.float32),
                                  y_true=K.constant(label_enc_fake, K.tf.float32)))

    print("Loss = ",loss)

    show(img, labels=[label_true], colors=[COLOR_GREEN], name='True')
    show(img_crop, labels=[label_crop], colors=[COLOR_GREEN], name='Crop')
    show(img_crop, labels=[label_dec, label_dec_fake], colors=[COLOR_RED,COLOR_BLUE], name='Decoded',legend=LEGEND_CORNERS)
