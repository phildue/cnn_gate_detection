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
name = "office"
shot_path = "resource/samples/" + name + "/"

n_samples = 10
batch_size = 10
cam_range_side = (-3, 3)
cam_range_forward = (0, -30)
cam_range_lift = (-0.5, 1.5)
cam_range_pitch = (-0.1, 0.1)
cam_range_roll = (-0.1+np.pi, 0.1+np.pi)
cam_range_yaw = (-0.1+np.pi, -0.1+np.pi)

n_light_range = (4, 6)
n_gate_range = (2, 3)

gate_pos_range_z = (-5, 5)
gate_pos_range_x = (-4, 4)

width, height = (640, 640)
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
set_writer = SetFileParser(shot_path, img_format='bmp', label_format='xml', start_idx=0)
n_batches = int(n_samples / batch_size)
for i in range(n_batches):
    tic()
    samples, labels = samplegen.generate(n_samples=batch_size)
    for n in range(len(samples)):
        show(samples[n],labels=labels[n])
    set_writer.write(samples, labels)

    toc("Batch: {0:d}/{2:d}, {1:d} shots generated after ".format(i + 1, len(samples), n_batches))

set_analyzer = SetAnalyzer((height, width), shot_path)
set_analyzer.show_summary()
