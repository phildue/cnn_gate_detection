import os
from pathlib import Path

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History, TerminateOnNaN, LearningRateScheduler, \
    ReduceLROnPlateau, CSVLogger

from modelzoo.augmentation.AugmenterEnsemble import AugmenterEnsemble
from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator


class Training:
    def __init__(self,
                 predictor: Predictor, dataset_gen: DatasetGenerator, out_file, patience=3,
                 log_dir='./logs', stop_on_nan=True
                 , lr_schedule=None, lr_reduce=-1, log_csv=True, initial_epoch=0, epochs=100):
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.dataset_gen = dataset_gen
        self.predictor = predictor
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
            tensorboard = TensorBoard(batch_size=dataset_gen.batch_size, log_dir=log_dir, write_images=True,
                                      histogram_freq=0)
            callbacks.append(tensorboard)

        if stop_on_nan:
            stop_nan = TerminateOnNaN()
            callbacks.append(stop_nan)

        if lr_schedule is not None:
            schedule = LearningRateScheduler(schedule=lr_schedule)
            callbacks.append(schedule)

        if lr_reduce > -1:
            reducer = ReduceLROnPlateau(monitor='loss', factor=lr_reduce, patience=patience - 1, min_lr=0.00001)
            callbacks.append(reducer)

        if log_csv:
            log_file_name = log_dir + '/log.csv'
            append = Path(log_file_name).is_file()
            csv_logger = CSVLogger(log_file_name, append=append)
            callbacks.append(csv_logger)

        history = History()
        callbacks.append(history)

        self.callbacks = callbacks

    def fit_generator(self):

        history = self.predictor.net.backend.fit_generator(
            generator=self.predictor.preprocessor.preprocess_train_generator(self.dataset_gen.generate()),
            steps_per_epoch=(self.dataset_gen.n_samples / self.dataset_gen.batch_size),
            epochs=self.epochs,
            initial_epoch=self.initial_epoch,
            verbose=1,
            validation_data=self.predictor.preprocessor.preprocess_train_generator(self.dataset_gen.generate_valid()),
            validation_steps=100,
            callbacks=self.callbacks)

        return history.history

    @property
    def summary(self):

        if isinstance(self.predictor.preprocessor.augmenter, AugmenterEnsemble):
            augmentation = [augmenter.__class__.__name__ for augmenter in
                            self.predictor.preprocessor.augmenter.augmenters]
        else:
            augmentation = self.predictor.preprocessor.augmenter.__class__.__name__

        summary = {'model': self.predictor.net.__class__.__name__,
                   'resolution': self.predictor.input_shape,
                   'train_params': self.predictor.net.train_params,
                   'image_source': self.dataset_gen.source_dir,
                   'color_format': self.dataset_gen.color_format,
                   'batch_size': self.dataset_gen.batch_size,
                   'n_samples': self.dataset_gen.n_samples,
                   'augmentation': augmentation,
                   'initial_epoch': self.initial_epoch,
                   'epochs': self.epochs}
        return summary
