import keras.backend as K
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras.optimizers import Adam

from modelzoo.models.Preprocessor import Preprocessor
from modelzoo.models.cornernet.CornerNetEncoder import CornerNetEncoder
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs
from utils.workdir import cd_work

cd_work()
img_shape = (416, 416, 3)
batch_size = 8
log_dir = 'out/cornerdetect/'
create_dirs([log_dir])
netin = Input(img_shape)

net = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(netin)
net = LeakyReLU()(net)
net = BatchNormalization()(net)

net = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(net)
net = LeakyReLU()(net)
net = BatchNormalization()(net)

net = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(net)
net = LeakyReLU()(net)
net = BatchNormalization()(net)

net = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(net)
net = LeakyReLU()(net)
net = BatchNormalization()(net)

net = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(net)
net = LeakyReLU()(net)
net = BatchNormalization()(net)
net = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(net)

model = Model(netin, net)


def segment_loss(y_true, y_pred):
    positives = y_true[:, :, :]
    loss = K.binary_crossentropy(target=y_true, output=y_pred, from_logits=True)
    loss_weighted = (positives * loss * 25000 + (1 - positives) * loss)
    return K.mean(K.mean(loss_weighted, -1), -1)


def edge_error(y_true, y_pred):
    y_true = K.reshape(y_true, (batch_size, -1))
    y_pred = K.reshape(y_pred, (batch_size, -1))
    edge = y_true
    predicted_edge = K.cast(K.sigmoid(y_pred) >= 0.5, K.floatx())
    diff = K.clip(edge - predicted_edge, 0, 1.0)
    n_wrong = K.sum(diff, -1)
    n_total = K.sum(edge, -1)
    error = n_wrong / n_total
    return error


def background_error(y_true, y_pred):
    y_true = K.reshape(y_true, (batch_size, -1))
    y_pred = K.reshape(y_pred, (batch_size, -1))
    background = 1 - y_true
    predicted_bg = K.cast(K.sigmoid(y_pred) < 0.5, K.floatx())
    diff = K.clip(background - predicted_bg, 0, 1.0)
    n_wrong = K.sum(diff, -1)
    n_total = K.sum(background, -1)
    error = n_wrong / n_total
    return error


model.compile(
    loss=segment_loss,
    optimizer=Adam(0.001, 0.9, 0.999, 1e-08, 0.0005),
    metrics=[edge_error, background_error],

)

# model = load_model('out/cornerdetect/model.h5', custom_objects={'segment_loss': segment_loss,
#                                                                 'edge_error': edge_error,
#                                                                 'background_error': background_error})

dataset_loader = GateGenerator(['resource/ext/samples/iros2018_flights/'], batch_size=batch_size, valid_frac=0.1,
                               color_format='bgr', label_format='xml', n_samples=None,
                               remove_filtered=False, max_empty=0, filter=None)

encoder = CornerNetEncoder(img_shape[:2])

preprocessor = Preprocessor(augmenter=None, encoder=encoder, n_classes=1, img_shape=img_shape[:2], color_format='bgr')

log_file_name = log_dir + '/log.csv'

model.fit_generator(
    generator=preprocessor.preprocess_train_generator(dataset_loader.generate()),
    steps_per_epoch=int(dataset_loader.n_samples / dataset_loader.batch_size),
    initial_epoch=0,
    epochs=50,
    validation_data=preprocessor.preprocess_train_generator(dataset_loader.generate_valid()),
    validation_steps=10,
    callbacks=[
        EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min',
                      verbose=1),
        ModelCheckpoint(log_dir + 'model.h5', monitor='val_loss', verbose=1,
                        save_best_only=True,
                        mode='min', save_weights_only=False,
                        period=1),
        CSVLogger(log_file_name, append=False)]

)
