# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import numpy as np
import os

import time

from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.airsim.AirSimGen import AirSimGen
from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from samplegen.shotgen.positiongen.RandomPositionGen import RandomPositionGen
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.Imageprocessing import show
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc
from utils.workdir import cd_work

cd_work()

# TODO choose simulation environment here + camera settings and start simulation


client = AirSimClient()
while True:
    sample, label = client.retrieve_samples()
    time.sleep(2)
    show(sample, labels=label, t=1)
    filtered = []
    # for o in label.objects:
    #     if 0 < o.position.yaw < 20.0 or \
    #         160 < o.position.yaw < 180:
    #         pass
    #     else:
    #         filtered.append(o)
    # l_filtered = ImgLabel(filtered)
    # show(sample, labels=l_filtered, t=1, name='filtered')
