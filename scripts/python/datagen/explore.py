import os
import sys

import numpy as np
from shotgen.engine3d.opengl.GateGLTall import GateGLTall

from shotgen.engine3d.opengl.GateGLThin250 import GateGLThin250

from workdir import work_dir

from shotgen.engine3d.opengl.GateGLLarge import GateGLThickLarge

from fileaccess.SetFileParser import SetFileParser
from labels.Pose import Pose
from shotgen.GateGen import GateGen
from shotgen.engine3d.Explorer import Explorer
from shotgen.engine3d.opengl.GateGLOpen import GateGLOpen
from shotgen.positiongen.RandomPositionGen import RandomPositionGen

work_dir()

from scene.Camera import Camera
from scene.GateThin250 import Gate250
from scene.Light import Light
from scene.Scene import Scene
from shotgen.engine3d.SceneEngine import SceneEngine
from fileaccess.utils import save
from shotgen.engine3d.opengl.OpenGlView import OpenGlView

cam = Camera(1000, init_pose=Pose(dist_forward=5))

gate_path = "resource/gates/"
gate_file = "gate250.obj"

gate_1 = (GateGLOpen(), Pose().to_scene_unit)
gate_2 = (GateGLThickLarge(), Pose(dist_side=-2.0, dist_forward=2.0, yaw=np.pi / 4).to_scene_unit)

gate_generator = GateGen(gates=[GateGLThickLarge(), GateGLTall()], n_gate_range=(3, 8))

# gates = [gate_1]
gates = gate_generator.generate()
scene_engine = SceneEngine(Scene(cam, objects=gates, lights=[Light((-2, 2, 3)), Light((0, 2, 4)), Light((-2, 0, 3))]))

cam_range_side = (-1, 1)
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
recorder = None  # SetFileParser(shot_path, img_format='bmp', label_format='pkl', start_idx=0)
explorer = Explorer(scene_engine, position_gen=position_gen, recorder=recorder)
explorer.event_loop()
# save(explorer.trajectory, 'recorded_trajectroy.pkl', shot_path)
# 3.341|Side-dist:-0.221|Lift:0.883|
# Roll:-0.434|Pitch:-0.264|Yaw:-0.334|