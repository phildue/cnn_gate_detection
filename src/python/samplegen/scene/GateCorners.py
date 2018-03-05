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
    def as_mat(self) -> np.array:
        return np.array([
            self.center,
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,

        ])
