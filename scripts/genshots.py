import os
import sys

import numpy as np

from SetAnalyzer import SetAnalyzer
from shotgen.GateGen import GateGen
from timing import tic, tuc, toc

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from fileaccess.SetFileParser import SetFileParser
from scene.Scene import Scene
from shotgen.ShotCreate import ShotCreate
from shotgen.engine3d.SceneEngine import SceneEngine

from shotgen.lightgen.RandomLightGen import RandomLightGen
from shotgen.positiongen.RandomPositionGen import RandomPositionGen

name = "mult_gate_aligned"
shot_path = "samplegen/resource/shots/" + name + "/"

n_positions = 1000
n_batches = 10
cam_range_side = (-0.5, 0.5)
cam_range_forward = (0, 20)
cam_range_lift = (-0.5, 1.0)
cam_range_pitch = (-0.1, 0.1)
cam_range_roll = (-0.1, 0.1)
cam_range_yaw = (-np.pi / 2, np.pi / 2)
light_range_x = (-12, 1)
light_range_y = (-12, 1)
light_range_z = (4, 5)
n_light_range = (6, 6)
n_gate_range = (2, 4)

gate_pos_range_z = (1, 15)
gate_pos_range_x = (-2, 2)

gate_path = "samplegen/resource/gates/"
gate_file = "gate250.obj"
width, height = (640, 640)

gate_gen = GateGen(gate_path=gate_path, gate_file=gate_file, n_gate_range=n_gate_range, forw_gate_range=gate_pos_range_z
                   , side_gate_range=gate_pos_range_x, min_gate_dist=2)

position_gen = RandomPositionGen(range_dist_side=cam_range_side,
                                 range_dist_forward=cam_range_forward,
                                 range_lift=cam_range_lift,
                                 range_pitch=cam_range_pitch,
                                 range_roll=cam_range_roll,
                                 range_yaw=cam_range_yaw)

light_gen = RandomLightGen(
    range_side=light_range_x,
    range_lift=light_range_y,
    range_forward=light_range_z,
    n_light_range=n_light_range
)

scene = Scene()
scene_engine = SceneEngine(scene, width=width, height=height)
shot_creator = ShotCreate(position_gen, light_gen, scene_engine, perc_empty=0.05)

if not os.path.exists(shot_path):
    os.makedirs(shot_path)
setwriter = SetFileParser(shot_path, img_format='bmp', label_format='pkl', start_idx=0)

for i in range(n_batches):
    tic()
    scene.objects = gate_gen.generate()
    shots, labels = shot_creator.get_shots(int(n_positions / n_batches))

    setwriter.write(shots, labels)

    toc("Batch: {0:d}/{2:d}, {1:d} shots generated after ".format(i + 1, len(shots), n_batches))

scene_engine.stop()

SetAnalyzer((width, height), shot_path).get_heat_map().show()
