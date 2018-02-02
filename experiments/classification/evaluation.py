import os

import sys
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

testset = 'resource/samples/classif2/test/'
test_datagen = ImageDataGenerator(rescale=1. / 255)
classifier = load_model('dvlab/resource/models/1CNN.h5')

test_set = test_datagen.flow_from_directory(testset,
                                            target_size=(48, 48),
                                            batch_size=32,
                                            class_mode='binary')

score = classifier.evaluate_generator(test_set, steps=(400 / 32))
print("Loss: ", score[0], "Accuracy: ", score[1])