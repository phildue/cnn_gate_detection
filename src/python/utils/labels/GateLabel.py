import numpy as np

from utils.labels.GateCorners import GateCorners
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Pose import Pose


class GateLabel(ObjectLabel):
    # TODO add "shift" that moves the points

    def __init__(self, pose: Pose = None, gate_corners: GateCorners = None, confidence=1.0, class_name='gate'):
        self.gate_corners = gate_corners
        x_min = min([self.gate_corners.top_left[0], self.gate_corners.top_right[0], self.gate_corners.bottom_left[0],
                     self.gate_corners.bottom_right[0], self.gate_corners.center[0]])
        y_min = min([self.gate_corners.top_left[1], self.gate_corners.top_right[1], self.gate_corners.bottom_left[1],
                     self.gate_corners.bottom_right[1], self.gate_corners.center[1]])
        x_max = max([self.gate_corners.top_left[0], self.gate_corners.top_right[0], self.gate_corners.bottom_left[0],
                     self.gate_corners.bottom_right[0], self.gate_corners.center[0]])
        y_max = max([self.gate_corners.top_left[1], self.gate_corners.top_right[1], self.gate_corners.bottom_left[1],
                     self.gate_corners.bottom_right[1], self.gate_corners.center[1]])
        box = np.array([[x_min, y_min], [x_max, y_max]])
        super().__init__(class_name, box, confidence)
        self.pose = pose

    @property
    def csv(self):
        return "{0:.3f},{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:03d},{7:03d},{8:03d},{9:03d},{10:03d},{11:03d}," \
               "{12:03d},{13:03d}" \
            .format(self.pose.dist_x, self.pose.dist_y,
                    self.pose.up, self.pose.pitch,
                    self.pose.roll,
                    self.pose.yaw,
                    self.gate_corners.top_right[0],
                    self.gate_corners.top_right[1],
                    self.gate_corners.top_left[0],
                    self.gate_corners.top_left[1],
                    self.gate_corners.bottom_right[0],
                    self.gate_corners.bottom_right[1],
                    self.gate_corners.bottom_left[0],
                    self.gate_corners.bottom_left[1])
