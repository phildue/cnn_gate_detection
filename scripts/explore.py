import os
import sys

import numpy as np

from fileaccess.SetFileParser import SetFileParser
from labels.Pose import Pose
from shotgen.GateGen import GateGen
from shotgen.engine3d.Explorer import Explorer
from shotgen.positiongen.RandomPositionGen import RandomPositionGen

PROJECT_ROOT = '/home/phil/Desktop/thesis/code/dronevision'

WORK_DIRS = [PROJECT_ROOT + '/samplegen/src/python',
             PROJECT_ROOT + '/droneutils/src/python',
             PROJECT_ROOT + '/dvlab/src/python']
for work_dir in WORK_DIRS:
    sys.path.insert(0, work_dir)
os.chdir(PROJECT_ROOT)

from scene.Camera import Camera
from scene.Gate250 import Gate250
from scene.Light import Light
from scene.Scene import Scene
from shotgen.engine3d.SceneEngine import SceneEngine
from fileaccess.utils import save
from shotgen.engine3d.opengl.OpenGlView import OpenGlView

cam = Camera(1000, init_pose=Pose(dist_forward=15))

gate_path = "samplegen/resource/gates/"
gate_file = "gate250.obj"
gate_1 = (OpenGlView(Gate250(), gate_path, gate_file), Pose(yaw=np.pi/4))
gate_2 = (
    OpenGlView(Gate250(), gate_path, gate_file), Pose(dist_side=-2.0, dist_forward=2.0, yaw=np.pi / 4))

gate_generator = GateGen(gate_path=gate_path, gate_file=gate_file, n_gate_range=(3, 4))

gates = [gate_1]
# gates = gate_generator.generate()
scene_engine = SceneEngine(Scene(cam, objects=gates,lights=[Light((-2, 2, 3)), Light((0, 2, 4)), Light((-2, 0, 3))]))

cam_range_side = (-0.5, 0.5)
cam_range_forward = (0, 10)
cam_range_lift = (-0.5, 1.0)
cam_range_pitch = (-np.pi / 4, np.pi / 4)
cam_range_roll = (-np.pi / 4, np.pi / 4)
cam_range_yaw = (-np.pi, np.pi)

n_gate_range = (2, 4)
gate_pos_range = (0, 10)
position_gen = RandomPositionGen(range_dist_side=cam_range_side,
                                 range_dist_forward=cam_range_forward,
                                 range_lift=cam_range_lift,
                                 range_pitch=cam_range_pitch,
                                 range_roll=cam_range_roll,
                                 range_yaw=cam_range_yaw)

shot_path = 'samplegen/resource/shots/stream_recorded/'
if not os.path.exists(shot_path):
    os.makedirs(shot_path)
recorder = SetFileParser(shot_path, img_format='bmp', label_format='pkl', start_idx=0)
explorer = Explorer(scene_engine, position_gen=position_gen, recorder=recorder)
explorer.event_loop()
save(explorer.trajectory, 'recorded_trajectroy.pkl', shot_path)
# 3.341|Side-dist:-0.221|Lift:0.883|
# Roll:-0.434|Pitch:-0.264|Yaw:-0.334|
