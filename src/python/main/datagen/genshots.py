import os

import numpy as np

from samplegen.SetAnalyzer import SetAnalyzer
from samplegen.scene.Scene import Scene
from samplegen.shotgen.GateGen import GateGen
from samplegen.shotgen.ShotCreate import ShotCreate
from samplegen.shotgen.engine3d.SceneEngine import SceneEngine
from samplegen.shotgen.engine3d.opengl.GateGLLarge import GateGLThickLarge
from samplegen.shotgen.engine3d.opengl.GateGLTall import GateGLTall
from samplegen.shotgen.engine3d.opengl.GateGLThin250 import GateGLThin250
from samplegen.shotgen.lightgen.RandomLightGen import RandomLightGen
from samplegen.shotgen.positiongen.RandomPositionGen import RandomPositionGen
from utils.fileaccess.SetFileParser import SetFileParser
from utils.fileaccess.utils import create_dirs
from utils.timing import tic, toc
from utils.workdir import work_dir

work_dir()

name = "mult_gate_aligned"
shot_path = "resource/shots/" + name + "/"

N = 99
n_positions = 10000 - N * 100
n_batches = 100 - N
cam_range_side = (-1, 1)
cam_range_forward = (-5, 5)
cam_range_lift = (-0.5, 1.0)
cam_range_pitch = (-0.1, 0.1)
cam_range_roll = (-0.1, 0.1)
cam_range_yaw = (-np.pi, np.pi)
light_range_x = (-4, 4)
light_range_y = (-6, 6)
light_range_z = (4, 5)
n_light_range = (4, 6)
n_gate_range = (2, 3)

gate_pos_range_z = (-8, 8)
gate_pos_range_x = (-3, 3)

width, height = (640, 640)

gate_gen = GateGen(gates=[GateGLTall(), GateGLThin250(), GateGLThickLarge()], n_gate_range=n_gate_range,
                   forw_gate_range=gate_pos_range_z
                   , side_gate_range=gate_pos_range_x, min_gate_dist=(2, 2))

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

create_dirs([shot_path])
setwriter = SetFileParser(shot_path, img_format='bmp', label_format='pkl', start_idx=N * 100)

for i in range(n_batches):
    tic()
    scene.objects = gate_gen.generate()
    shots, labels = shot_creator.get_shots(int(n_positions / n_batches))

    setwriter.write(shots, labels)

    toc("Batch: {0:d}/{2:d}, {1:d} shots generated after ".format(i + 1, len(shots), n_batches))

scene_engine.stop()

SetAnalyzer((width, height), shot_path).get_heat_map().show()
