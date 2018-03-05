import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.Backend import resize, crop


def sample_image(arr, idx0, idx1):
    res = np.zeros((3,))
    if idx0 < 0 or idx1 < 0 or idx0 > (arr.shape[0] - 1) or idx1 > (arr.shape[1] - 1):
        return res
    idx0_fl = int(np.floor(idx0))
    idx0_cl = int(np.ceil(idx0))
    idx1_fl = int(np.floor(idx1))
    idx1_cl = int(np.ceil(idx1))

    s1 = arr[idx0_fl, idx1_fl]
    s2 = arr[idx0_fl, idx1_cl]
    s3 = arr[idx0_cl, idx1_cl]
    s4 = arr[idx0_cl, idx1_fl]
    x = idx0 - idx0_fl
    y = idx1 - idx1_fl
    res[0] = s1[0] * (1 - x) * (1 - y) + s2[0] * (1 - x) * y + s3[0] * x * y + s4[0] * x * (1 - y)
    res[1] = s1[1] * (1 - x) * (1 - y) + s2[1] * (1 - x) * y + s3[1] * x * y + s4[1] * x * (1 - y)
    res[2] = s1[2] * (1 - x) * (1 - y) + s2[2] * (1 - x) * y + s3[2] * x * y + s4[2] * x * (1 - y)

    return res.T


def calc_shift(x1, x2, cx, k):
    thresh = 1
    x3 = x1 + (x2 - x1) * 0.5
    res1 = x1 + ((x1 - cx) * k * ((x1 - cx) ** 2))
    res3 = x3 + ((x3 - cx) * k * ((x3 - cx) ** 2))

    if -thresh < res1 < thresh:
        return x1
    if res3 < 0:
        return calc_shift(x3, x2, cx, k)
    else:
        return calc_shift(x1, x3, cx, k)


def calc_rad(x, y, center, k, scale, shift):
    xshift, yshift = shift
    x_scale, y_scale = scale
    cx, cy = center
    x = (x * x_scale + xshift)
    x = np.reshape(x, (1, -1))
    x = np.tile(x, (len(y), 1))
    y = (y * y_scale + yshift)
    y = np.reshape(y, (-1, 1))
    result_y = y + ((y - cy) * k * ((x - cx) ** 2 + (y - cy) ** 2))
    result_x = x + ((x - cx) * k * ((x - cx) ** 2 + (y - cy) ** 2))

    return result_y, result_x


def fisheye(img: Image, k, label=None, center=None):
    img_eye, label_eye = resize(img, (300, 300), label=label)

    if center is None:
        cx, cy = img_eye.shape[1] / 2, img_eye.shape[0] / 2
    else:
        cx, cy = center

    # mat = cv2.cvtColor(img.array, cv2.COLOR_BGR2RGBA)
    mat = img_eye.array
    h, w = img_eye.shape[:2]
    x_shift = calc_shift(0, cx - 1, cx, k)
    new_center_x = w - cx
    x_shift_2 = calc_shift(0, new_center_x - 1, new_center_x, k)

    y_shift = calc_shift(0, cy - 1, cy, k)
    new_center_y = w - cy
    y_shift_2 = calc_shift(0, new_center_y - 1, new_center_y, k)

    x_scale = (w - x_shift - x_shift_2) / w
    y_scale = (h - y_shift - y_shift_2) / h

    y = np.arange(0, h)
    x = np.arange(0, w)
    y_rad, x_rad = calc_rad(x, y, (cx, cy), k, (x_scale, y_scale), (x_shift, y_shift))

    img_eye = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            # print('(%f|%f)' % (y_rad, x_rad))
            img_eye[i, j] = sample_image(mat, y_rad[i, j], x_rad[i, j])

    # img_eye = cv2.cvtColor(img_eye, cv2.COLOR_RGBA2BGR)
    # img_eye /= 255.0
    img_eye = img_eye.astype(np.uint8)

    img_eye, label_eye = crop(Image(img_eye, img.format), (30, 30), (h - 30, w - 30), label_eye)
    img_eye, label_eye = resize(img_eye, shape=img.shape[:2], label=label_eye)

    return img_eye, label_eye
