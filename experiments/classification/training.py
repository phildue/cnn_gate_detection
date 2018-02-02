import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from setup import init_workspace

init_workspace()

trainset = 'resource/samples/classif/training/'
testset = 'resource/samples/classif/test/'
model_name = '1CNN_test'

from backend.models.OneCNN import one_cnn

classifier = one_cnn()

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(trainset,
                                                 target_size=(48, 48),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(testset,
                                            target_size=(48, 48),
                                            batch_size=32,
                                            class_mode='binary')

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

classifier.fit_generator(training_set,
                         steps_per_epoch=int(np.round((8000 / 32))),
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=int(np.round(2000 / 32)),
                         callbacks=[early_stop, checkpoint])

classifier.save(model_name + '.hdf5')
