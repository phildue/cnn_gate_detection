import numpy as np


class Pose:
    METER_2_SCENE_UNIT = 1.167

    def __init__(self, north=0.0, east=0.0, up=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        self.pitch = pitch
        self.roll = roll
        self.lift = up
        self.east = east
        self.north = north
        self.yaw = yaw

    def __repr__(self):
        return "|Forward-dist:{0:.3f}|Side-dist:{1:.3f}|Lift:{2:.3f}|\nRoll:{3:.3f}|Pitch:{4:.3f}|Yaw:{5:.3f}|".format(
            self.north,
            self.east,
            self.lift, self.roll,
            self.pitch,
            self.yaw)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.pitch == other.pitch and \
               self.roll == other.roll and \
               self.yaw == other.yaw and \
               self.north == other.dist_forward and \
               self.east == other.dist_side and \
               self.lift == other.lift

    def __add__(self, other):
        return Pose(north=self.north + other.north,
                    east=self.east + other.east,
                    up=self.lift + other.up,
                    yaw=self.yaw + other.yaw,
                    roll=self.roll + other.roll,
                    pitch=self.pitch + other.pitch, )

    def __sub__(self, other):
        return Pose(north=self.north - other.north,
                    east=self.east - other.east,
                    up=self.lift - other.up,
                    yaw=self.yaw - other.yaw,
                    roll=self.roll - other.roll,
                    pitch=self.pitch - other.pitch, )

    @property
    def rotmat_pitch(self):
        return np.array([[1, 0, 0],
                         [0, np.cos(self.pitch), -np.sin(self.pitch)],
                         [0, np.sin(self.pitch), np.cos(self.pitch)]
                         ])

    @property
    def rotmat_roll(self):
        return np.array([[np.cos(self.roll), -np.sin(self.roll), 0],
                         [np.sin(self.roll), np.cos(self.roll), 0],
                         [0, 0, 1]
                         ])

    @property
    def rotmat_yaw(self):
        return np.array([[np.cos(self.yaw), 0, np.sin(self.yaw)],
                         [0, 1, 0],
                         [-np.sin(self.yaw), 0, np.cos(self.yaw)]
                         ])

    @property
    def transfmat(self):
        transformation_mat = np.eye(4)
        transformation_mat[:3, :3] = self.rotmat
        transformation_mat[:3, 3] = self.transvec
        return transformation_mat

    @property
    def transvec(self):
        return np.array([self.east, self.lift, self.north]).T

    @transvec.setter
    def transvec(self, translation):
        self.east = translation[0]
        self.lift = translation[1]
        self.north = translation[2]

    @property
    def rotmat(self):
        return self.rotmat_pitch.dot(self.rotmat_yaw).dot(self.rotmat_roll)


    @property
    def to_scene_unit(self):
        return Pose(self.north * self.METER_2_SCENE_UNIT,
                    self.east * self.METER_2_SCENE_UNIT,
                    self.lift * self.METER_2_SCENE_UNIT,
                    self.roll,
                    self.pitch,
                    self.yaw)

    @property
    def to_meters(self):
        return Pose(self.north / self.METER_2_SCENE_UNIT,
                    self.east / self.METER_2_SCENE_UNIT,
                    self.lift / self.METER_2_SCENE_UNIT,
                    self.roll,
                    self.pitch,
                    self.yaw)
