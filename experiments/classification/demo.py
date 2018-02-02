import glob
import os
import random
import sys
import timeit

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from imageprocessing.Backend import show, imread


gate_files = glob.glob('resource/samples/classif2/test/gate/*.jpg')
nogate_files = glob.glob('resource/samples/classif2/test/nogate/*.jpg')
classifier = load_model('dvlab/resource/models/1CNN.h5')

for i in range(50):
    sample = random.choice(random.choice([gate_files,nogate_files]))

    test_image = image.load_img(sample, target_size=(48, 48))
    test_image = image.img_to_array(test_image)
    test_image_vec = np.expand_dims(test_image, axis=0)
    tic = timeit.default_timer()
    result = classifier.predict(test_image_vec)
    toc = timeit.default_timer()
    if result[0][0] == 1:
        prediction = 'nogate'
    else:
        prediction = 'gate'

    print(prediction)
    print("Prediction time: {0:f} seconds".format(toc-tic))
    show(imread(sample), "sample")
