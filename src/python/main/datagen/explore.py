import os

import numpy as np
from workdir import work_dir

from src.python.samplegen.shotgen import Explorer
from src.python.samplegen.shotgen import GateGLOpen
from src.python.samplegen.shotgen import GateGLTall
from src.python.samplegen.shotgen import GateGLThickLarge
from src.python.samplegen.shotgen import GateGen
from src.python.samplegen.shotgen import RandomPositionGen
from src.python.utils.labels.Pose import Pose

work_dir()

from src.python.samplegen.scene import Camera
from src.python.samplegen.scene import Scene
from src.python.samplegen.shotgen import SceneEngine

cam = Camera(1000, init_pose=Pose(dist_forward=5))

gate_path = "resource/gates/"
gate_file = "gate250.obj"

gate_1 = (GateGLOpen(), Pose().to_scene_unit)
gate_2 = (GateGLThickLarge(), Pose().to_scene_unit)

gate_generator = GateGen(gates=[GateGLThickLarge(), GateGLTall()], n_gate_range=(3, 8))

gates = [gate_2]
#gates = gate_generator.generate()
scene_engine = SceneEngine(Scene(cam, objects=gates))

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
