from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History

from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator


def fit_generator(predictor: Predictor, dataset_gen: DatasetGenerator, out_file, batch_size, initial_epoch=0,
                  epochs=100, patience=3, log_dir='./logs'):
    callbacks = []
    if patience > -1:
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, mode='min', verbose=1)
        callbacks.append(early_stop)
    if out_file is not None:
        checkpoint = ModelCheckpoint(log_dir + out_file, monitor='val_loss', verbose=2, save_best_only=True,
                                     mode='min', save_weights_only=False,
                                     period=1)
        callbacks.append(checkpoint)
    if log_dir is not None:
        tensorboard = TensorBoard(batch_size=batch_size, log_dir=log_dir, write_images=True, histogram_freq=0)
        callbacks.append(tensorboard)

    history = History()
    callbacks.append(history)

    predictor.net.backend.fit_generator(
        generator=predictor.preprocessor.preprocess_train_generator(dataset_gen.generate()),
        steps_per_epoch=(dataset_gen.n_samples / batch_size),
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        validation_data=predictor.preprocessor.preprocess_train_generator(dataset_gen.generate_valid()),
        validation_steps=100,
        callbacks=callbacks)

    return history.history
