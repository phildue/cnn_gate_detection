import numpy as np


class GateCorners:
    def __init__(self, center: tuple,
                 top_left: tuple,
                 top_right: tuple,
                 bottom_right: tuple,
                 bottom_left: tuple):
        self.center = center
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.top_right = top_right
        self.top_left = top_left

    @property
    def mat(self) -> np.array:
        return np.array([

            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
            self.center,
        ])

    @staticmethod
    def from_mat(mat: np.array):
        return GateCorners(center=mat[0],
                           top_left=mat[1],
                           top_right=mat[2],
                           bottom_right=mat[3],
                           bottom_left=mat[4])
