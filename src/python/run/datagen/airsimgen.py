# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import numpy as np

from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.airsim.AirSimGen import AirSimGen
from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from samplegen.shotgen.positiongen.RandomPositionGen import RandomPositionGen
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.imageprocessing.Imageprocessing import show
from utils.timing import tic, toc
from utils.workdir import cd_work

cd_work()
name = "brick_room"
shot_path = "resource/ext/samples/" + name + "/"

n_samples = 2000
batch_size = 100
cam_range_side = (-3, 3)
cam_range_forward = (-8, 8)
cam_range_lift = (0.5, 2.0)
cam_range_pitch = (-0.3, 0.3)
cam_range_roll = (-0.3, 0.3)
cam_range_yaw = (-np.pi, np.pi)

#TODO choose simulation environment here + camera settings and start simulation

posegen = RandomPositionGen(range_dist_side=cam_range_side,
                            range_dist_forward=cam_range_forward,
                            range_lift=cam_range_lift,
                            range_pitch=cam_range_pitch,
                            range_roll=cam_range_roll,
                            range_yaw=cam_range_yaw)

client = AirSimClient()
samplegen = AirSimGen(posegen, client)

create_dirs([shot_path])
set_writer = SetFileParser(shot_path, img_format='jpg', label_format='xml', start_idx=0)
n_batches = int(n_samples / batch_size)
for i in range(n_batches):
    tic()
    samples, labels = samplegen.generate(n_samples=batch_size)

    set_writer.write(samples, labels)

    toc("Batch: {0:d}/{2:d}, {1:d} shots generated after ".format(i + 1, len(samples), n_batches))

set_analyzer = SetAnalyzer((640, 480), shot_path)
set_analyzer.show_summary()
