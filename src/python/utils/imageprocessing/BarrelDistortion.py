import numpy as np
from cv2 import morphologyEx, MORPH_CLOSE

from utils.imageprocessing.Image import Image
from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc


class BarrelDistortion:
    def __init__(self, img_shape, rad_dist_params, squeeze=1.0, non_rad_dist_params=(0, 0),
                 max_iterations=100, distortion_radius=1.0, center=None, conv_thresh=0.01):
        self.img_shape = img_shape
        self.epsilon = conv_thresh
        self.center = center
        self.distortion_radius = distortion_radius
        self.non_rad_dist_params = non_rad_dist_params
        self.max_iterations = max_iterations
        self.squeeze = squeeze
        self.rad_dist_params = rad_dist_params
        self.mapping_undist, self.mapping_dist = self.create_mapping()

    def create_mapping(self):
        h, w = self.img_shape
        x = np.tile(np.arange(0, w), (h, 1))
        y = np.tile(np.reshape(np.arange(0, h), (h, 1)), (1, w))
        coords = np.concatenate((np.expand_dims(x, -1), np.expand_dims(y, -1)), -1)
        coords_norm = self._normalize(coords.astype(np.float))
        mapping_undist = self._distortion_model(coords_norm)
        mapping_dist = 2 * coords_norm - mapping_undist
        # mapping_dist = self._inverse_model(coords_norm,mapping_dist)
        mapping_undist = self._denormalize(mapping_undist)
        mapping_dist = self._denormalize(mapping_dist)
        return mapping_undist.astype(np.uint), mapping_dist.astype(np.uint)

    def _distortion_model(self, mat: np.array):
        k_1, k_2 = self.rad_dist_params
        l_y, l_x = self.non_rad_dist_params
        s = self.squeeze
        x = mat[:, :, 0]
        y = mat[:, :, 1]

        x_d = x * (1 + k_1 * x ** 2 + k_1 * (1 + l_x) * y ** 2 + k_2 * (x ** 2 + y ** 2) ** 2)
        y_d = y * (1 + k_1 / s * x ** 2 + k_1 / s * (1 + l_y) * y ** 2 + k_2 / s * (x ** 2 + y ** 2) ** 2)
        mat_d = np.concatenate((np.expand_dims(x_d, -1), np.expand_dims(y_d, -1)), -1)

        return mat_d

    def _normalize(self, mat: np.array):
        """
        Normalizes the pixel coordinates so that the origin is center and the radius is distortion_radius.
        :param mat: coordinate matrix
        :return: normalize coordinate matrix
        """
        h, w = mat.shape[:2]

        center = self.center if self.center is not None else np.array([w, h]) / 2

        h /= self.distortion_radius
        w /= self.distortion_radius

        mat_dimension_less = (mat - center) / np.linalg.norm(np.array([w, h]) / 2)
        return mat_dimension_less

    def _denormalize(self, mat: np.array):

        h, w = mat.shape[:2]
        center = self.center if self.center is not None else np.array([w, h]) / 2

        h /= self.distortion_radius
        w /= self.distortion_radius

        mat_denorm = mat * np.linalg.norm(np.array([w, h]) / 2) + center

        return mat_denorm

    def undistort(self, img: Image, label: ImgLabel = None):
        mat = img.array
        mat_undistorted = self._undistort_mat(mat)
        # mat_undistorted = self._fill_holes(mat_undistorted)
        # label_distorted = self._distort_label(label, center)

        return Image(mat_undistorted, img.format), label

    def _fill_holes(self, mat):
        h, w = self.img_shape
        mat_filled = mat.copy()
        iterations = 8
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                for _ in range(iterations):
                    if np.all(mat_filled[i, j] == 0):
                        mat_filled[i, j, 0] = np.mean(mat_filled[i - 1:i + 1, j - 1:j + 1, 0])
                        mat_filled[i, j, 1] = np.mean(mat_filled[i - 1:i + 1, j - 1:j + 1, 1])
                        mat_filled[i, j, 2] = np.mean(mat_filled[i - 1:i + 1, j - 1:j + 1, 2])
        return mat_filled

    def distort(self, img: Image, label: ImgLabel = None):
        mat = img.array
        mat_undistorted = self._distort_mat(mat)
        # mat_undistorted = morphologyEx(mat_undistorted, MORPH_CLOSE, (3, 3), iterations=2)

        return Image(mat_undistorted, img.format), label

    def _undistort_mat(self, mat):

        h, w = self.img_shape
        mat_undistorted = np.zeros_like(mat)
        for i in range(h):
            for j in range(w):
                if 0 <= self.mapping_undist[i, j][1] < h and 0 <= self.mapping_undist[i, j][0] < w:
                    mat_undistorted[self.mapping_undist[i, j][1], self.mapping_undist[i, j][0]] = mat[i, j]

        return mat_undistorted

    def _distort_mat(self, mat):

        h, w = self.img_shape
        mat_distorted = np.zeros_like(mat)
        for i in range(h):
            for j in range(w):
                if 0 <= self.mapping_dist[i, j][1] < h and 0 <= self.mapping_dist[i, j][0] < w:
                    mat_distorted[self.mapping_dist[i, j][1], self.mapping_dist[i, j][0]] = mat[i, j]

        return mat_distorted

    def _distort_label(self, label: ImgLabel, center):
        label_distorted = label.copy()
        for obj in label_distorted.objects:
            x_min_d, y_min_d = self.distort_point(np.array(obj.x_min, obj.y_min), self.k, center)
            x_max_d, y_max_d = self.distort_point(np.array(obj.x_max, obj.y_max), self.k, center)
            obj.x_min = x_min_d
            obj.x_max = x_max_d
            obj.y_min = y_min_d
            obj.y_max = y_max_d
            if isinstance(obj, GateLabel):
                corners = obj.gate_corners.as_mat
                corners_dist = np.apply_along_axis(self.distort_point, 0, corners,
                                                   kwargs={'k': self.k, 'center': center})
                obj.gate_corners = GateCorners.from_mat(corners_dist)

        return label_distorted

    def _inverse_model(self, mat: np.array, init):
        h, w = mat.shape[:2]
        mat_cur = init.copy().astype(np.float)
        for t in range(self.max_iterations):
            tic()
            mat_prev = mat_cur.copy()
            dist_model = self._distortion_model(mat_prev)
            gradient = self._gradient(mat_prev)
            for i in range(h):
                for j in range(w):
                    grad = gradient[i, j]
                    np.fill_diagonal(grad, np.diag(grad) + 0.000001)
                    prev = np.reshape(mat_prev[i, j], (2, 1))
                    diff = np.reshape((dist_model[i, j] - mat[i, j]), (2, 1))
                    update = prev - 0.1 * np.dot(np.linalg.inv(grad), diff)
                    mat_cur[i, j] = update.flatten()
            delta = np.linalg.norm(mat_cur - mat_prev)
            if delta < self.epsilon:
                break
            toc('iteration: {} - delta: {} - time: '.format(t, np.round(delta, 2)))
        return mat_cur

    def _gradient(self, mat):
        k_1, k_2 = self.rad_dist_params
        l_y, l_x = self.non_rad_dist_params
        s = self.squeeze

        x = mat[:, :, 0]
        y = mat[:, :, 1]
        h, w = mat.shape[:2]

        dx_x = k_1 * x ** 2 + k_1 * y ** 2 * (l_x + 1) + k_2 * (x ** 2 + y ** 2) ** 2 + x * (
            2 * k_1 * x + 4 * k_2 * x * (x ** 2 + y ** 2)) + 1
        dx_y = x * (2 * k_1 * y * (l_x + 1) + 4 * k_2 * y * (x ** 2 + y ** 2))

        dy_x = y * (2 * k_1 * x / s + 4 * k_2 * x * (x ** 2 + y ** 2) / s)
        dy_y = k_1 * x ** 2 / s + k_1 * y ** 2 * (l_y + 1) / s + k_2 * (x ** 2 + y ** 2) ** 2 / s + y * (
            2 * k_1 * y * (l_y + 1) / s + 4 * k_2 * y * (x ** 2 + y ** 2) / s) + 1

        gradient = np.zeros((h, w, 2, 2))
        gradient[:, :, 0, 0] = dx_x
        gradient[:, :, 0, 1] = dx_y
        gradient[:, :, 1, 0] = dy_x
        gradient[:, :, 1, 1] = dy_y
        return gradient
