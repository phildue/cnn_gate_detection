import numpy as np


class Pose:
    METER_2_SCENE_UNIT = 1.167

    def __init__(self, dist_forward=0.0, dist_side=0.0, lift=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        self.pitch = pitch
        self.roll = roll
        self.lift = lift
        self.dist_side = dist_side
        self.dist_forward = dist_forward
        self.yaw = yaw

    def __repr__(self):
        return "|Forward-dist:{0:.3f}|Side-dist:{1:.3f}|Lift:{2:.3f}|\nRoll:{3:.3f}|Pitch:{4:.3f}|Yaw:{5:.3f}|".format(
            self.dist_forward,
            self.dist_side,
            self.lift, self.roll,
            self.pitch,
            self.yaw)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.pitch == other.pitch and \
               self.roll == other.roll and \
               self.yaw == other.yaw and \
               self.dist_forward == other.dist_forward and \
               self.dist_side == other.dist_side and \
               self.lift == other.lift

    def __add__(self, other):
        return Pose(dist_forward=self.dist_forward + other.dist_forward,
                    dist_side=self.dist_side + other.dist_side,
                    lift=self.lift + other.lift,
                    yaw=self.yaw + other.yaw,
                    roll=self.roll + other.roll,
                    pitch=self.pitch + other.pitch, )

    def __sub__(self, other):
        return Pose(dist_forward=self.dist_forward - other.dist_forward,
                    dist_side=self.dist_side - other.dist_side,
                    lift=self.lift - other.lift,
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
        return np.array([self.dist_side, self.lift, self.dist_forward]).T

    @transvec.setter
    def transvec(self, translation):
        self.dist_side = translation[0]
        self.lift = translation[1]
        self.dist_forward = translation[2]

    @property
    def rotmat(self):
        return np.matmul(np.matmul(self.rotmat_pitch, self.rotmat_yaw), self.rotmat_roll)

    @property
    def to_scene_unit(self):
        return Pose(self.dist_forward * self.METER_2_SCENE_UNIT,
                    self.dist_side * self.METER_2_SCENE_UNIT,
                    self.lift * self.METER_2_SCENE_UNIT,
                    self.roll,
                    self.pitch,
                    self.yaw)

    @property
    def to_meters(self):
        return Pose(self.dist_forward / self.METER_2_SCENE_UNIT,
                    self.dist_side / self.METER_2_SCENE_UNIT,
                    self.lift / self.METER_2_SCENE_UNIT,
                    self.roll,
                    self.pitch,
                    self.yaw)
