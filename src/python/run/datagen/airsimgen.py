# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
import sys
import cv2
from math import pi

import numpy as np
from cv2.cv2 import circle

import utils
from samplegen.scene.Camera import Camera
from utils.workdir import cd_work

sys.path.extend(["c:/Users/mail-/Documents/code/dronerace2018/target/simulator/AirSim/PythonClient"])
from AirSimClient import *

cd_work()
client = MultirotorClient()
client.confirmConnection()

found = client.simSetSegmentationObjectID("[\w]*", 0, True);
print("Done: %r" % (found))

found = client.simSetSegmentationObjectID("bottomright", 4)
found = client.simSetSegmentationObjectID("bottomleft", 1)
found = client.simSetSegmentationObjectID("topleft", 2)
found = client.simSetSegmentationObjectID("topright", 3)

responses = client.simGetImages([
    ImageRequest(0, AirSimImageType.Segmentation, False, False),  # scene vision image in uncompressed RGBA array
    ImageRequest(0, AirSimImageType.Scene, False, False)])
print('Retrieved images: %d', len(responses))


def response2numpy(response):
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgba = img1d.reshape(response.height, response.width, 4)  # reshape array to 4 channel image array H X W X 4
    # img_rgba = np.flipud(img_rgba)  # original image is fliped vertically
    return img_rgba


seg = response2numpy(responses[0])
scene = response2numpy(responses[1])
hcam, wcam = scene.shape[:2]
hseg, wseg = seg.shape[:2]
hscale = hcam / hseg
wscale = wcam / wseg
bottomright = np.where(np.all(seg == [190, 225, 64, 255], -1))
bottomright = np.mean(bottomright, -1)
bottomleft = np.where(np.all(seg == [153, 108, 6, 255], -1))
bottomleft = np.mean(bottomleft, -1)
topleft = np.where(np.all(seg == [112, 105, 191, 255], -1))
topleft = np.mean(topleft, -1)
topright = np.where(np.all(seg == [89, 121, 72, 255], -1))
topright = np.mean(topright, -1)
print(bottomright)
print(bottomleft)
print(topleft)
print(topright)
annotated = scene.copy()
cv2.circle(annotated, (int(bottomright[1] * wscale), int(bottomright[0] * hscale)), 5, (0, 255, 0))
cv2.circle(annotated, (int(bottomleft[1] * wscale), int(bottomleft[0] * hscale)), 5, (0, 255, 0))
cv2.circle(annotated, (int(topright[1] * wscale), int(topright[0] * hscale)), 5, (0, 255, 0))
cv2.circle(annotated, (int(topleft[1] * wscale), int(topleft[0] * hscale)), 5, (0, 255, 0))
cv2.imshow("Segmentation", seg)
cv2.imshow("Scene", scene)
cv2.imshow("Annotated", annotated)

cv2.waitKey(0)
