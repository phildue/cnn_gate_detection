import os
import sys

import numpy as np
from workdir import work_dir

from shotgen.engine3d.opengl.GateGLTall import GateGLTall

from labels.Pose import Pose
from scene.Light import Light
from shotgen.lightgen.ConstantLightGen import ConstantLightGen
from shotgen.positiongen.TrajectoryGen import TrajectoryGen
from timing import tic, toc

work_dir()

from fileaccess.SetFileParser import write_set
from scene.GateThin250 import Gate250
from scene.Scene import Scene
from shotgen.ShotCreate import ShotCreate
from shotgen.engine3d.SceneEngine import SceneEngine
from shotgen.engine3d.opengl.OpenGlView import OpenGlView

name = "stream_3"
shot_path = "resource/shots/" + name + "/"


def yaw_circle(dist_forward, steps):
    waypoints = []
    for i in range(steps):
        waypoints.append(Pose(dist_forward=dist_forward,
                              dist_side=0,
                              lift=0,
                              roll=0,
                              pitch=0,
                              yaw=i * 2 * np.pi / steps))
    return waypoints


def pitch_half_circle(dist_forward, steps):
    waypoints = []
    for i in range(steps):
        waypoints.append(Pose(dist_forward=dist_forward,
                              dist_side=0,
                              lift=0,
                              roll=0,
                              pitch=-i * 2 * np.pi / (2 * steps),
                              yaw=0))
    return waypoints


def roll_half_circle(dist_forward, steps):
    waypoints = []
    for i in range(steps):
        waypoints.append(Pose(dist_forward=dist_forward,
                              dist_side=0,
                              lift=0,
                              pitch=0,
                              roll=-i * 2 * np.pi / (2 * steps),
                              yaw=0))
    return waypoints


def roll_yaw_move(dist_forward, steps):
    waypoints = []
    for i in range(steps):
        waypoints.append(Pose(dist_forward=dist_forward,
                              dist_side=0,
                              lift=0,
                              roll=0,
                              pitch=-i * 2 * np.pi / (2 * steps),
                              yaw=i * 2 * np.pi / (2 * steps)))
    return waypoints


def approach(start_dist, end_dist, steps, yaw, pitch, roll):
    waypoints = []

    # approach
    for i in range(steps_per_move):
        waypoints.append(Pose(start_dist - abs(start_dist - end_dist) / steps * i, 0, 0, roll, pitch, yaw))
    return waypoints


def genshots(n_positions=500,
             output_path=None,
             waypoints=None
             ):
    gate_path = "samplegen/resource/gates/"
    gate_file = "gate250.obj"
    position_gen = TrajectoryGen(waypoints)

    light_gen = ConstantLightGen([Light((-2, 2, 3)), Light((0, 2, 4)), Light((-2, 0, 3))])

    tic()

    scene_engine = SceneEngine(Scene(objects=[(GateGLTall(), Pose())]), width=640,
                               height=640)
    shots, shot_labels = ShotCreate(position_gen, light_gen, scene_engine).get_shots(n_positions)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_set(output_path, shots, shot_labels, img_format='bmp')

    toc(str(len(shots)) + " shots generated in: ")


steps_per_move = 100
trajectory = []

trajectory.extend(approach(30, 0, steps_per_move, yaw=0, pitch=0, roll=0))
trajectory.extend(approach(30, 0, steps_per_move, yaw=45, pitch=0, roll=0))
trajectory.extend(yaw_circle(30, steps_per_move))
trajectory.extend(yaw_circle(20, steps_per_move))
trajectory.extend(yaw_circle(10, steps_per_move))
trajectory.extend(yaw_circle(5, steps_per_move))
# trajectory.extend(pitch_half_circle(30, steps_per_move))
# trajectory.extend(pitch_half_circle(20, steps_per_move))
# trajectory.extend(pitch_half_circle(10, steps_per_move))
# trajectory.extend(pitch_half_circle(5, steps_per_move))
# trajectory.extend(roll_half_circle(30, steps_per_move))
# trajectory.extend(roll_half_circle(20, steps_per_move))
# trajectory.extend(roll_half_circle(10, steps_per_move))
# trajectory.extend(roll_half_circle(5, steps_per_move))

genshots(waypoints=trajectory,
         output_path='samplegen/resource/shots/stream/',
         n_positions=len(trajectory))
