import cv2
import numpy as np
from keras.models import load_model

from utils.imageprocessing.Backend import imread, imshow, resize
from utils.imageprocessing.Image import Image
from utils.workdir import cd_work

cd_work()

detector = load_model('out/cornerdetect_gray/model.h5', compile=False)

# for i in range(1, 2):
#     model.layers[i].set_weights(weights[i])

img = imread('resource/ext/samples/daylight_course1/00055.jpg', 'bgr')
img.array = cv2.cvtColor(img.array, cv2.COLOR_BGR2GRAY)
img.array = np.expand_dims(img.array, -1)
mat = np.expand_dims(img.array, 0)
netout = detector.predict(mat)[0]
netout_scaled = (netout - np.min(netout)) / np.ptp(netout) * 255
imshow(img, 'org')
for i in range(netout_scaled.shape[2]):
    activations = Image(netout_scaled[:, :, i].astype(np.uint8), 'bgr')
    activations = resize(activations, (416, 416))
    imshow(activations, str(i))
