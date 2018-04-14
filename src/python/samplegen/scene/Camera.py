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

    def get_rel_pose(self, obj_pose: Pose):
        """
        Calculates the pose of an object towards the camera.
        :param obj_pose: pose of the object in world space
        :return: pose: pose of the object within camera space
        """
        # TODO move this to some more global config
        obj_center = np.array([[0],
                               [0],
                               [0]])

        obj_up = np.array([[0],
                           [1],
                           [0]])
        obj_east = np.array([[1],
                             [0],
                             [0]])

        cam_north = np.array([[0],
                              [0],
                              [1]])
        cam_east = np.array([[1],
                             [0],
                             [0]])

        obj2world = obj_pose.transfmat
        world2cam = np.linalg.inv(self.pose.transfmat)

        obj_hom = np.vstack([np.hstack([obj_center, obj_up, obj_east]),
                             [[1, 1, 1]]])
        obj_world = obj2world.dot(obj_hom)
        obj_cam = world2cam.dot(obj_world)

        obj_cam_center = obj_cam[:3, 0]
        obj_cam_up = obj_cam[:3, 1]
        obj_cam_east = obj_cam[:3, 2]
        plane_pitch = obj_cam_up - obj_cam_center
        plane_yaw = obj_cam_east - obj_cam_center

        yaw_cam = np.math.acos(plane_yaw.dot(cam_north) / (np.linalg.norm(plane_yaw) * np.linalg.norm(cam_north)))
        pitch_cam = np.math.acos(plane_pitch.dot(cam_north) / (np.linalg.norm(plane_pitch) * np.linalg.norm(cam_north)))
        roll_cam = np.math.acos(plane_yaw.dot(cam_east) / (np.linalg.norm(plane_yaw) * np.linalg.norm(cam_east)))

        return Pose(north=obj_cam_center[2], east=obj_cam_center[0], up=obj_cam_center[1], yaw=yaw_cam,
                    pitch=pitch_cam, roll=roll_cam)
