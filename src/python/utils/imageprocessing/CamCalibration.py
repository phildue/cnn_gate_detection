# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import cv2
import numpy as np


class CamCalibration:
    def __init__(self, img_shape, img_type='chess'):

        self.img_type = img_type
        self.img_shape = img_shape

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = None, None, None, None, None

    def calibrate(self, images):
        obj_points, img_points = self._find_points(images)
        if len(obj_points) > 0 and len(img_points) > 0:
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self._get_cam_params(obj_points, img_points)
            return self.calc_estimation_error(obj_points, img_points)
        else:
            print('Failure.No points found.')
            return 1.0

    def demo(self, img):
        img = cv2.imread(img)
        h, w = img.shape[:2]
        cam_mat_new, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, self.mtx, self.dist, None, cam_mat_new)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imshow('calibresult', dst)

    def calc_estimation_error(self, obj_points, img_points):
        mean_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error / len(obj_points)
        return mean_error

    def save(self, path):
        with open(path, 'wb') as f:
            np.savetxt(f, self.ret, delimiter=',', newline='\r\n', header='Ret')
            np.savetxt(f, self.mtx, delimiter=',', newline='\r\n', header='Mtx')
            np.savetxt(f, self.dist, delimiter=',', newline='\r\n', header='Dist')
            np.savetxt(f, self.rvecs, delimiter=',', newline='\r\n', header='rvecs')
            np.savetxt(f, self.tvecs, delimiter=',', newline='\r\n', header='tvecs')

    def _find_points(self, images):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        for img in images:
            gray = cv2.cvtColor(img.array, cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray', gray)
            cv2.waitKey(0)
            if self.img_type is 'chess':
                ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            else:
                ret, corners = cv2.findCirclesGrid(gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        return obj_points, img_points

    def _get_cam_params(self, obj_points, img_points):
        return cv2.calibrateCamera(obj_points, img_points, self.img_shape[::-1], None, None)
