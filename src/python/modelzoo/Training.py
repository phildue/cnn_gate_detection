from pathlib import Path

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History, TerminateOnNaN, LearningRateScheduler, \
    ReduceLROnPlateau, CSVLogger
from markdown.preprocessors import Preprocessor

from modelzoo.models.Encoder import Encoder
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble


class Training:
    def __init__(self,
                 encoder: Encoder,
                 preprocessor: Preprocessor,
                 dataset_gen: DatasetGenerator,
                 out_file,
                 input_shape,
                 patience_early_stop=3,
                 patience_lr_reduce=2,
                 log_dir='./logs',
                 stop_on_nan=True,
                 lr_schedule=None,
                 lr_reduce=-1,
                 log_csv=True,
                 initial_epoch=0,
                 epochs=100,
                 callbacks=None,
                 validation_generator: DatasetGenerator = None):
        self.input_shape = input_shape
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.validation_generator = validation_generator if validation_generator is not None else dataset_gen.generate_valid()
        self.patience_lr_reduce = patience_lr_reduce
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.dataset_gen = dataset_gen
        self.callbacks = []
        if patience_early_stop > -1:
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience_early_stop, mode='min',
                                       verbose=1)
            self.callbacks.append(early_stop)
        if out_file is not None:
            checkpoint = ModelCheckpoint(log_dir + out_file, monitor='val_loss', verbose=1,
                                         save_best_only=True,
                                         mode='min', save_weights_only=False,
                                         period=1)
            self.callbacks.append(checkpoint)
        if log_dir is not None:
            tensorboard = TensorBoard(batch_size=dataset_gen.batch_size, log_dir=log_dir, write_images=True,
                                      histogram_freq=0)
            self.callbacks.append(tensorboard)

        if stop_on_nan:
            stop_nan = TerminateOnNaN()
            self.callbacks.append(stop_nan)

        if lr_schedule is not None:
            schedule = LearningRateScheduler(schedule=lr_schedule)
            self.callbacks.append(schedule)

        if lr_reduce > -1:
            reducer = ReduceLROnPlateau(monitor='loss', factor=lr_reduce, patience=patience_lr_reduce, min_lr=0.00001)
            self.callbacks.append(reducer)

        if log_csv:
            log_file_name = log_dir + '/log.csv'
            append = Path(log_file_name).is_file() and initial_epoch > 0
            csv_logger = CSVLogger(log_file_name, append=append)
            self.callbacks.append(csv_logger)
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        history = History()
        self.callbacks.append(history)

    def fit_generator(self):

        history = self.model.fit_generator(
            generator=self.preprocessor.preprocess_train_generator(self.dataset_gen.generate()),
            steps_per_epoch=(self.dataset_gen.n_samples / self.dataset_gen.batch_size),
            epochs=self.epochs,
            initial_epoch=self.initial_epoch,
            verbose=1,
            validation_data=self.preprocessor.preprocess_train_generator(self.validation_generator),
            validation_steps=100,
            callbacks=self.callbacks)

        return history.history

    @property
    def summary(self):

        if isinstance(self.preprocessor.augmenter, RandomEnsemble):
            augmentation = ''
            augmenters = self.preprocessor.augmenter.augmenters
            probs = self.preprocessor.augmenter.probs
            for i in range(len(augmenters)):
                augmentation += '\n{0:.2f} -> {1:s}'.format(probs[i], augmenters[i].__class__.__name__)
        else:
            augmentation = self.preprocessor.augmenter.__class__.__name__

        summary = {'resolution': self.input_shape,
                   'image_source': self.dataset_gen.source_dir,
                   'color_format': self.dataset_gen.color_format,
                   'batch_size': self.dataset_gen.batch_size,
                   'n_samples': self.dataset_gen.n_samples,
                   'transform': augmentation,
                   'initial_epoch': self.initial_epoch,
                   'epochs': self.epochs,
                   # 'architecture': self.predictor.net.backend.get_config(),
                   'weights': self.modelcount_params()}
        return summary
