import numpy as np

from utils.fileaccess.utils import load_file, save_file
from utils.imageprocessing.Image import Image
from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc
from utils.imageprocessing.DistortionModel import DistortionModel


class BarrelDistortion(DistortionModel):
    @staticmethod
    def from_file(mapping_file: str):
        return load_file(mapping_file)

    def save(self, filename='barrel_dist.pkl'):
        save_file(self, filename, './')

    def __init__(self, img_shape, rad_dist_params, squeeze=1.0, tangential_dist_params=(0, 0),
                 max_iterations=100, distortion_radius=1.0, center=None, conv_thresh=0.01, scale=1.0):
        """
        Barrel Distortion model.
        [Vass, G., & Perlaki, T. (n.d.). Applying and removing lens distortion in post production.
        Retrieved from http://www.vassg.hu/pdf/vass_gg_2003_lo.pdf]

        :param img_shape: original image shape
        :param rad_dist_params: parameters for radial distortion
        :param squeeze: parameter for squeeze effect
        :param tangential_dist_params: parameters for non-radial distortion (yshift, xshift)
        :param distortion_radius: radius on which distortion should be applied
        :param center: center around which distortion is applied
        :param scale: scale distortion
        :param conv_thresh: convergence threshold for newton approximation
        :param max_iterations: maximum iterations for newton approximation
        """
        self.scale = scale
        self.img_shape = img_shape
        self.epsilon = conv_thresh
        self.center = center
        self.distortion_radius = distortion_radius
        self.non_rad_dist_params = tangential_dist_params
        self.max_iterations = max_iterations
        self.squeeze = squeeze
        self.rad_dist_params = rad_dist_params
        self.mapping_undist, self.mapping_dist = self._create_mapping()

    def undistort(self, img: Image, label: ImgLabel = None):
        """
        Remove distortion from an image
        :param img: image
        :param label: label
        :return: img,label without distortion
        """
        mat = img.array
        mat_undistorted = self._distort_mat(mat, self.mapping_dist)
        if label is not None:
            label_undistorted = self._distort_label(label, self.mapping_undist)
            return Image(mat_undistorted, img.format), label_undistorted
        else:
            return Image(mat_undistorted, img.format)

    def distort(self, img: Image, label: ImgLabel = None):
        """
        Apply distortion to an image
        :param img: image
        :param label:  label
        :return: img, label with distortion
        """
        mat = img.array
        mat_undistorted = self._distort_mat(mat, self.mapping_undist)
        if label is not None:
            label_distorted = self._distort_label(label, self.mapping_dist)
            return Image(mat_undistorted, img.format), label_distorted
        else:
            return Image(mat_undistorted, img.format)

    @staticmethod
    def _distort_label(label: ImgLabel, mapping):
        """
        Move the coordinates of the label according to distortion.
        :param label: image label
        :param mapping: mapping containing new pixel coordinates given the old coordinates
        :return: distorted label
        """
        label_distorted = label.copy()
        h, w = mapping.shape[:2]
        for obj in label_distorted.objects:
            y_min = np.max([obj.y_min, 1])
            x_min = np.max([obj.x_min, 1])
            y_max = np.min([obj.y_max, h - 1])
            x_max = np.min([obj.x_max, w - 1])
            x_min_d, y_min_d = mapping[h - int(y_min), int(x_min)]
            x_max_d, y_max_d = mapping[h - int(y_max), int(x_max)]
            obj.x_min = np.min([x_min_d, x_max_d])
            obj.x_max = np.max([x_min_d, x_max_d])
            obj.y_min = h - np.min([y_min_d, y_max_d])
            obj.y_max = h - np.max([y_min_d, y_max_d])
            if isinstance(obj, GateLabel):
                corners = obj.gate_corners.mat
                corners_dist = np.zeros_like(corners)
                for i in range(corners.shape[0]):
                    corners_dist[i] = mapping[int(corners[i, 1]), int(corners[i, 0])]
                obj.gate_corners = GateCorners.from_mat(corners_dist)

        return label_distorted

    def _create_mapping(self):
        """
        Creates mapping to apply and remove distortion.
        :return:  mapping to apply and remove distortion
        """
        h, w = self.img_shape
        x = np.tile(np.arange(0, w), (h, 1))
        y = np.tile(np.reshape(np.arange(0, h), (h, 1)), (1, w))
        coords = np.concatenate((np.expand_dims(x, -1), np.expand_dims(y, -1)), -1)
        coords_norm = self._normalize(coords.astype(np.float))
        mapping_undist = self.scale * self._distortion_model(coords_norm)
        mapping_dist = 1 / self.scale * self._inverse_model_approx(coords_norm, np.zeros_like(coords_norm))
        mapping_undist = self._denormalize(mapping_undist)
        mapping_dist = self._denormalize(mapping_dist)
        return mapping_undist, mapping_dist

    def _distortion_model(self, coord: np.array):
        """
        Model for distortion.:
        :param coord: distorted pixel coordinates
        :return: undistored pixel coordinates
        """
        k_1, k_2 = self.rad_dist_params
        l_y, l_x = self.non_rad_dist_params
        s = self.squeeze
        x = coord[:, :, 0]
        y = coord[:, :, 1]

        x_d = x * (1 + k_1 * x ** 2 + k_1 * (1 + l_x) * y ** 2 + k_2 * (x ** 2 + y ** 2) ** 2)
        y_d = y * (1 + k_1 / s * x ** 2 + k_1 / s * (1 + l_y) * y ** 2 + k_2 / s * (x ** 2 + y ** 2) ** 2)
        mat_d = np.concatenate((np.expand_dims(x_d, -1), np.expand_dims(y_d, -1)), -1)

        return mat_d

    def _normalize(self, coord: np.array):
        """
        Normalizes the pixel coordinates so that the origin is center and the radius is distortion_radius.
        :param coord: coordinate matrix
        :return: normalize coordinate matrix
        """
        h, w = coord.shape[:2]

        center = self.center if self.center is not None else np.array([w, h]) / 2

        h /= self.distortion_radius
        w /= self.distortion_radius

        mat_dimension_less = (coord - center) / np.linalg.norm(np.array([w, h]) / 2)
        return mat_dimension_less

    def _denormalize(self, coord: np.array):
        """
        Scales the normalized pixel coordinates back to original size.
        :param coord: normalized pixel coordinates
        :return: denormalized coordinates
        """
        h, w = coord.shape[:2]
        center = self.center if self.center is not None else np.array([w, h]) / 2

        h /= self.distortion_radius
        w /= self.distortion_radius

        mat_denorm = coord * np.linalg.norm(np.array([w, h]) / 2) + center

        return mat_denorm

    @staticmethod
    def _bilinear_interp(mat: np.array, x, y):
        """
        Approximates f(x,y) using f(0,0),f(0,1),f(1,0),f(1,1) that is
        the new image given the 4 neighbouring pixels of the source image
        :param mat: source image
        :param x: source x
        :param y: source y
        :return: interpolated new pixel
        """
        channels = mat.shape[-1]
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = int(np.ceil(x))
        y1 = int(np.ceil(y))
        f00 = mat[y0, x0]
        f01 = mat[y1, x0]
        f10 = mat[y0, x1]
        f11 = mat[y1, x1]
        f = np.array([[f00, f01],
                      [f10, f11]])

        y_ = y - float(y0)
        x_ = x - float(x0)
        y_vec = np.array([[1 - y_], [y_]])
        x_vec = np.array([[1 - x_, x_]])
        out = np.zeros((channels,), dtype=np.int8)
        for i in range(channels):
            out[i] = (x_vec.dot(f[:, :, i].dot(y_vec))).astype(np.int8)

        return out.T

    def _distort_mat(self, mat: np.array, mapping: np.array):
        """
        Distort the image with bilinear interpolation.
        :param mat: image
        :param mapping: mapping containing the old indeces given the new coordinates
        :return: distorted image
        """
        h, w = self.img_shape
        mat_distorted = np.zeros_like(mat)
        for i in range(h):
            for j in range(w):
                x, y = mapping[i, j]
                if 1.0 < x < float(w) - 1.0 and 1.0 < y < float(h) - 1.0:
                    mat_distorted[i, j] = self._bilinear_interp(mat, x, y)

        return mat_distorted

    def _inverse_model_approx(self, coord: np.array, init):
        """
        Approximates the inverted model to apply distortion with newton method
        :param coord: pixel coordinates
        :param init: initial guess for mapping
        :return: approximated pixel coordinates after distortion
        """
        h, w = coord.shape[:2]
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
                    diff = np.reshape((dist_model[i, j] - coord[i, j]), (2, 1))
                    update = prev - np.linalg.inv(grad).dot(diff)
                    mat_cur[i, j] = update.flatten()
            delta = np.linalg.norm(mat_cur - mat_prev)
            if delta < self.epsilon:
                break
            toc('iteration: {} - delta: {} - time: '.format(t, np.round(delta, 2)))
        return mat_cur

    def _gradient(self, coord: np.array):
        """
        Gradient of distortion model
        :param coord: pixel coordinates
        :return: value of gradient
        """
        k_1, k_2 = self.rad_dist_params
        l_y, l_x = self.non_rad_dist_params
        s = self.squeeze

        x = coord[:, :, 0]
        y = coord[:, :, 1]
        h, w = coord.shape[:2]

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
