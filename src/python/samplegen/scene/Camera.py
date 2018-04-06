"""
Chapter 4.1, Programming Computer Vision with Python, Jan Erik Solem
"""
import numpy as np

from utils.labels.Pose import Pose


class Camera:
    def __init__(self, focal_length_x: float, focal_length_y: float, skew: float = 0.0, alpha: float = 1.0,
                 camera_center: (int, int) = (0, 0),
                 init_pose=Pose()):
        """
        :param focal_length_x:
        :param focal_length_y: distance between image plane and camera center, controls magnification/angle of view, the higher the focal length
                             the stronger the magnification -> the smaller the angle of view
        :param skew: use if the pixel array in the sensor is skewed
        :param alpha: used if pixels are non square
        :param camera_center: camera center on image
        :param offsets: zero rotation and translation of camera (roll, yaw pitch,lift,dist_forward,dist_side)
        in most cases its safe to assume default parameters, only focal_length needs to be calibrated
        """

        self.calibration_mat = np.array([[alpha * focal_length_x, skew, camera_center[0]],
                                         [0, focal_length_y, camera_center[1]],
                                         [0, 0, 1]])

        self.pose = init_pose

    def project(self, points: np.array) -> np.array:
        """
        :param points: n-by-4 matrix with points in 3d homogeneous coordinates
        :return: normalized projection of X on the image plane
        """
        projection_mat = np.matmul(self.calibration_mat, self.pose.transfmat[:3])
        projection = np.matmul(projection_mat, points.T)

        for i in range(3):
            projection[i] /= abs(projection[2])

        return projection.T
