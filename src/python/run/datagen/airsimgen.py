# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
import sys
import cv2
from math import pi

import numpy as np
from cv2.cv2 import circle, cvtColor, COLOR_RGB2BGR, COLOR_RGBA2BGR

import utils
from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.scene.Camera import Camera
from utils.imageprocessing.Imageprocessing import show, LEGEND_BOX
from utils.workdir import cd_work

sys.path.extend(["c:/Users/mail-/Documents/code/dronerace2018/target/simulator/AirSim/PythonClient"])
from AirSimClient import *

cd_work()
client = AirSimClient(n_gates=1)
img, label = client.retrieve_samples()

show(img, labels=label, legend=LEGEND_BOX)
