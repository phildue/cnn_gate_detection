import cv2
import numpy as np
from cv2.cv2 import remap

from utils.fileaccess.utils import load_file, save_file
from utils.imageprocessing.DistortionModel import DistortionModel
from utils.imageprocessing.Image import Image
from utils.labels.GateCorners import GateCorners
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc


class BarrelDistortion(DistortionModel):
    @staticmethod
    def from_file(mapping_file: str):
        return load_file(mapping_file)

    def save(self, filename='barrel_dist.pkl'):
        save_file(self, filename, './')

    def __init__(self, img_shape, rad_dist_params, squeeze=1.0, tangential_dist_params=(0, 0),
                 max_iterations=100, distortion_radius=1.0, center=None, conv_thresh=0.01):
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
        :param conv_thresh: convergence threshold for newton approximation
        :param max_iterations: maximum iterations for newton approximation
        """
        self.img_shape = img_shape
        self.epsilon = conv_thresh
        self.center = center
        self.distortion_radius = distortion_radius
        self.non_rad_dist_params = tangential_dist_params
        self.max_iterations = max_iterations
        self.squeeze = squeeze
        self.rad_dist_params = rad_dist_params
        self.map_u, self.map_d = self._create_mapping()

    def undistort(self, img: Image, label: ImgLabel = None, scale=1.0):
        """
        Remove distortion from an image
        :param img: image
        :param label: label
        :return: img,label without distortion
        """
        mat = img.array

        center = np.array([img.shape[1] / 2, img.shape[0] / 2])

        mapping_undist_c, mapping_dist_c = self._scale(center, scale)

        mat_undistorted = self._apply_mapping(mat, mapping_dist_c)
        if label is not None:
            label_undistorted = self._distort_label(label, mapping_undist_c)
            return Image(mat_undistorted, img.format), label_undistorted
        else:
            return Image(mat_undistorted, img.format)

    def distort(self, img: Image, label: ImgLabel = None, scale=1.0):
        """
        Apply distortion to an image
        :param img: image
        :param label:  label
        :return: img, label with distortion
        """
        mat = img.array
        center = np.array([img.shape[1] / 2, img.shape[0] / 2])

        mapping_undist_c, mapping_dist_c = self._scale(center, scale)

        mat_undistorted = self._apply_mapping(mat, mapping_undist_c)
        if label is not None:
            label_distorted = self._distort_label(label, mapping_dist_c)
            return Image(mat_undistorted, img.format), label_distorted
        else:
            return Image(mat_undistorted, img.format)

    def _scale(self, center, scale):
        map_d = self.map_d - center
        map_u = self.map_u - center
        map_d *= scale
        map_u *= scale
        map_d += center
        map_u += center
        return map_u, map_d

    @staticmethod
    def _distort_label(label: ImgLabel, mapping):
        """
        Move the coordinates of the label according to distortion.
        :param label: image label
        :param mapping: mapping containing new pixel coordinates given the old coordinates
        :return: distorted label
        """

        h, w = mapping.shape[:2]
        objects_distorted = []
        for obj in label.objects:
            if isinstance(obj, GateLabel):
                corners = obj.gate_corners.mat
                corners_dist = np.zeros_like(corners)
                out_of_view = False
                for i in range(corners.shape[0]):
                    try:
                        corners_dist[i] = mapping[int(corners[i, 1]), int(corners[i, 0])]
                    except IndexError:
                        out_of_view = True
                if not out_of_view:
                    obj_d = obj.copy()
                    obj_d.gate_corners = GateCorners.from_mat(corners_dist)
                    objects_distorted.append(obj_d)
            else:
                y_min = np.max([obj.y_min, 1])
                x_min = np.max([obj.x_min, 1])
                y_max = np.min([obj.y_max, h - 1])
                x_max = np.min([obj.x_max, w - 1])
                try:
                    x_min_d, y_min_d = mapping[h - int(y_min), int(x_min)]
                    x_max_d, y_max_d = mapping[h - int(y_max), int(x_max)]
                    x_min = np.min([x_min_d, x_max_d])
                    x_max = np.max([x_min_d, x_max_d])
                    y_min = h - np.min([y_min_d, y_max_d])
                    y_max = h - np.max([y_min_d, y_max_d])
                    obj_d = obj.copy()
                    obj_d.__bounding_box = np.array([[x_min, y_min],
                                                     [x_max, y_max]])

                except IndexError:
                    continue

        return ImgLabel(objects_distorted)

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
        map_u = self._model(coords_norm)
        map_d = self._inverse_model_approx(coords_norm, np.zeros_like(coords_norm))
        map_u = self._denormalize(map_u)
        map_d = self._denormalize(map_d)
        return map_u, map_d

    def _model(self, coord: np.array):
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

        x_u = x * (1 + k_1 * x ** 2 + k_1 * (1 + l_x) * y ** 2 + k_2 * (x ** 2 + y ** 2) ** 2)
        y_u = y * (1 + k_1 / s * x ** 2 + k_1 / s * (1 + l_y) * y ** 2 + k_2 / s * (x ** 2 + y ** 2) ** 2)
        mat_u = np.concatenate((np.expand_dims(x_u, -1), np.expand_dims(y_u, -1)), -1)

        return mat_u

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

    def _apply_mapping(self, mat: np.array, mapping: np.array):
        """
        Distort the image with bilinear interpolation.
        :param mat: image
        :param mapping: mapping containing the old indeces given the new coordinates
        :return: distorted image
        """
        mat_remaped = remap(mat, mapping[:, :, 0].astype(np.float32), mapping[:, :, 1].astype(np.float32),
                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return mat_remaped

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
            dist_model = self._model(mat_prev)
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

    def __repr__(self):
        return 'Shape: {}\n' \
               'Eps: {}\n' \
               'Center: {}\n' \
               'DistRadius: {}\n' \
               'NonRad Params: {}\n' \
               'MaxIterations: {}\n' \
               'Squeeze: {}\n' \
               'RadDist Params: {}\n'.format(
                                             self.img_shape,
                                             self.epsilon,
                                             self.center,
                                             self.distortion_radius,
                                             self.non_rad_dist_params,
                                             self.max_iterations,
                                             self.squeeze,
                                             self.rad_dist_params)
