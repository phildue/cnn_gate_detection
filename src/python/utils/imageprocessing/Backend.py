import copy

import cv2
import numpy as np
from PIL.Image import frombytes, FLIP_TOP_BOTTOM
from utils.imageprocessing.Image import Image
from utils.labels.GateLabel import GateLabel
from utils.labels.ImgLabel import ImgLabel
from utils.labels.utils import resize_label


def imread(path, color_format) -> Image:
    return Image(cv2.imread(path), color_format)


def imwrite(img: Image, path):
    cv2.imwrite(path, img.array)


def parse_screenshot(pixels, width, height) -> Image:
    image = frombytes("RGB", (width, height), pixels)
    image = image.transpose(FLIP_TOP_BOTTOM)
    return Image(__pil2cv(image), 'bgr')


def __pil2cv(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    return open_cv_image[:, :, ::-1].copy()


def replace_background(img: Image, background: Image, background_color=(0, 0, 0)) -> Image:
    img_cv = img.array
    background_cv = background.array
    background_cv = cv2.resize(background_cv, (img_cv.shape[1], img_cv.shape[0]))
    merged = background_cv.copy()
    merged[img_cv != background_color] = img_cv[img_cv != background_color]
    return Image(merged, img.format)


def draw_gate_corners(img: Image, label: GateLabel) -> Image:
    annotated_img = img.array.copy()
    corners = label.gate_corners.as_mat

    corners[:, 1] = img.shape[0] - corners[:, 1]

    for i in range(corners.shape[0]):
        if 0 < corners[i, 0] < img.shape[1] and 0 < corners[i, 1] < img.shape[0]:
            cv2.circle(annotated_img, tuple(corners[i, :]), 3,
                       (0, 0, 255), -1)

    return Image(annotated_img, img.format)


def flip_y(point: tuple, ymax) -> tuple:
    return point[0], ymax - point[1]


def draw_bounding_box(img: Image, p1: tuple, p2: tuple, color=(0, 255, 0), thickness=2) -> Image:
    return Image(
        cv2.rectangle(img.array.copy(), flip_y(p1, img.shape[0]), flip_y(p2, img.shape[0]), color, thickness=thickness),
        img.format)


def imshow(img: Image, name: str = "Img", t=0):
    cv2.imshow(name, img.array)
    cv2.waitKey(t)


def annotate_bounding_box(img: Image, label: ImgLabel, color=(0, 255, 0)) -> Image:
    img_ann = img.copy()

    for obj in [obj for obj in label.objects if obj is not None]:
        img_ann = draw_bounding_box(img_ann, (obj.x_min, obj.y_min), (obj.x_max, obj.y_max), color)
        cv2.putText(img_ann.array, obj.class_id,
                    flip_y((obj.x_max, obj.y_max), img_ann.shape[0]), 0,
                    1e-3 * img_ann.array.shape[0], color, 0)
    return img_ann


def annotate_text(text: str, img: Image, xy: tuple = (10, 10), color=(0, 255, 0), thickness=0):
    img_ann = img.copy()
    xy_flip = flip_y(xy, img_ann.shape[0])
    cv2.putText(img_ann.array, text,
                xy_flip, 0,
                1e-3 * img_ann.array.shape[0], color, thickness=thickness)
    return img_ann


def resize(img: Image, shape: tuple = None, scale_x=1.0, scale_y=1.0, label: ImgLabel = None) -> (Image, ImgLabel):
    shape_cv = shape if shape is not None else (0, 0)
    img_resized = cv2.resize(img.array.copy(), shape_cv, fx=scale_x, fy=scale_y)
    if label is not None:
        label_resized = resize_label(label, img.array.shape, shape, scale_x, scale_y)
        return Image(img_resized, img.format), label_resized
    else:
        return Image(img_resized, img.format)


def flip(img: Image, label: ImgLabel = None, flip_code=1) -> (Image, ImgLabel):
    flipped = cv2.flip(img.array.copy(), flip_code)

    if label is not None:
        label_flipped = copy.deepcopy(label)
        for obj in label_flipped.objects:
            xmin_new = img.array.shape[1] - obj.x_max
            xmax_new = img.array.shape[1] - obj.x_min
            obj.x_min = xmin_new
            obj.x_max = xmax_new
    else:
        label_flipped = None

    return Image(flipped, img.format), label_flipped


def translate(img: Image, shift_x, shift_y, label: ImgLabel) -> (Image, ImgLabel):
    content = img.array.copy()

    shift_y = 0 if shift_y < 0 else shift_y
    shift_y = img.shape[0] if shift_y > img.shape[0] else shift_y

    shift_x = 0 if shift_x < 0 else shift_x
    shift_x = img.shape[1] if shift_x > img.shape[1] else shift_x

    content = content[shift_y: (shift_y + img.shape[0]), shift_x: (shift_x + img.shape[1])]
    if label is not None:
        label_translated = copy.deepcopy(label)
        for obj in label_translated.objects:
            obj.x_min = obj.x_min - shift_x
            obj.x_max = obj.x_max - shift_x
    else:
        label_translated = None
    return Image(content, img.format), label_translated


def color_shift(img, t, label=None) -> (Image, ImgLabel):
    content = img.array
    content = content * (1 + t)
    content = content / (255. * 2.)
    return Image(content, img.format), copy.deepcopy(label)


def normalize(img: Image, min_range: float = 0.0, max_range: float = 1.0):
    return Image(
        cv2.normalize(img.array.astype('float'), None, alpha=min_range, beta=max_range, norm_type=cv2.NORM_MINMAX),
        img.format)


def blur(img: Image, size: (int, int), iterations=10):
    mat = img.array
    for i in range(iterations):
        mat = cv2.GaussianBlur(mat, size, 0.5)
    return Image(mat, img.format)


def noisy(img: Image, var=0.1, iterations=10):
    row, col, ch = img.array.shape
    mean = 0
    sigma = var ** 0.5
    noised = img.array
    for i in range(iterations):
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noised += gauss.astype(np.uint8)
    return Image(noised, img.format)


COLOR_BGR2YUV = cv2.COLOR_BGR2YUV
COLOR_YUV2BGR = cv2.COLOR_YUV2BGR
COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY


def convert_color(img: Image, code):
    img_array = cv2.cvtColor(img.array, code)
    if code is COLOR_BGR2GRAY:
        img_array = np.expand_dims(img_array, -1)
        img_array = np.tile(img_array, (1, 1, 3))
    return Image(img_array, img.format)


def histogram_eq(img: Image):
    """
    Perform histogram equalization on the input image.

    See https://en.wikipedia.org/wiki/Histogram_equalization.
    """

    img_eq = np.copy(img.array)

    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_RGB2HSV)

    img_eq[:, :, 2] = cv2.equalizeHist(img_eq[:, :, 2])

    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_HSV2RGB)

    return Image(img_eq, img.format)


def brightness(img: Image, min=0.5, max=2.0):
    """
    Randomly change the brightness of the input image.

    Protected against overflow.
    """
    hsv = cv2.cvtColor(img.array, cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min, max)

    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return Image(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), img.format)


def crop(img: Image, min_xy=(0, 0), max_xy=None, label: ImgLabel = None):
    if max_xy is None:
        max_xy = img.shape[1], img.shape[0]

    x_min, x_max = min_xy[0], max_xy[0]
    y_min, y_max = min_xy[1], max_xy[1]

    y_max_cv, y_min_cv = img.shape[0] - min_xy[1], img.shape[0] - max_xy[1]

    img_crop = Image(img.array[y_min_cv:y_max_cv, x_min:x_max], img.format)
    label_crop = copy.deepcopy(label)

    if label is not None:
        objs_crop = []
        for obj in label.objects:
            delta_x = x_min
            obj.x_min -= delta_x
            obj.x_max -= delta_x
            delta_y = y_min
            obj.y_min -= delta_y
            obj.y_max -= delta_y

            obj.x_min = max(0, obj.x_min)
            obj.x_max = min(img_crop.shape[1], obj.x_max)
            obj.y_min = max(0, obj.y_min)
            obj.y_max = min(img_crop.shape[0], obj.y_max)

            if obj.area >= 20:
                objs_crop.append(obj)
        label_crop = ImgLabel(objs_crop)

    return img_crop, label_crop


def split_video(src_file, out_dir):
    vidcap = cv2.VideoCapture(src_file)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite(out_dir + "/frame{0:05d}.jpg".format(count), image)  # save frame as JPEG file
        count += 1
