import numpy as np
from imageprocessing.Backend import draw_bounding_box, draw_gate_corners, imshow, annotate_text, imwrite
from labels.GateLabel import GateLabel
from labels.ImgLabel import ImgLabel
from labels.ObjectLabel import ObjectLabel
from labels.Pose import Pose

from src.python.samplegen.scene import GateCorners
from src.python.utils.imageprocessing.Image import Image

LEGEND_BOX = 0
LEGEND_CORNERS = 1
LEGEND_TEXT = 2
LEGEND_POSITION = 3

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)


def get_bounding_box(gate_corners: GateCorners):
    x_min = min([gate_corners.top_left[0], gate_corners.top_right[0], gate_corners.bottom_left[0],
                 gate_corners.bottom_right[0], gate_corners.center[0]])
    y_min = min([gate_corners.top_left[1], gate_corners.top_right[1], gate_corners.bottom_left[1],
                 gate_corners.bottom_right[1], gate_corners.center[1]])
    x_max = max([gate_corners.top_left[0], gate_corners.top_right[0], gate_corners.bottom_left[0],
                 gate_corners.bottom_right[0], gate_corners.center[0]])
    y_max = max([gate_corners.top_left[1], gate_corners.top_right[1], gate_corners.bottom_left[1],
                 gate_corners.bottom_right[1], gate_corners.center[1]])
    return (x_min, y_min), (x_max, y_max)


def annotate_gate(img: Image, label: ImgLabel, bounding_box=False) -> Image:
    img_ann = img.copy()

    for obj in label.objects:
        if obj is not None:
            img_ann = draw_gate_corners(img_ann, obj)
            if bounding_box:
                img_ann = draw_bounding_box(img_ann, (int(obj.x_max), int(obj.y_max)), (int(obj.x_min), int(obj.y_min)))
    return img_ann


def annotate_position(img: Image, pos: Pose, x, y, color=(0, 255, 0)):
    img_ann = img.copy()
    lookup = [pos.dist_forward, pos.dist_side, pos.lift, pos.roll, pos.pitch, pos.yaw]
    for i in range(len(lookup)):
        img_ann = annotate_text("{0:0.2f}".format(lookup[i]), img_ann,
                                (int(x), int(y - (i + 1) * 20)), color)
    return img_ann


def annotate_label(img: Image, label: ImgLabel, color=None, legend=LEGEND_POSITION, thickness=None) -> Image:
    img_ann = img.copy()

    if color is None:
        color = (0, 255, 0)

    for obj in label.objects:
        if obj is None: continue
        try:
            confidence = obj.confidence
        except AttributeError:
            confidence = 1.0
        if isinstance(obj, GateLabel):
            if legend >= LEGEND_CORNERS:
                img_ann = draw_gate_corners(img_ann, obj)
            if legend >= LEGEND_POSITION:
                img_ann = annotate_position(img_ann, obj.position, obj.x_min, obj.y_max + 5, color)
        if isinstance(obj, ObjectLabel):
            if thickness is None:
                thickness = int(np.round(confidence * 4, 0))

            img_ann = draw_bounding_box(img_ann, (int(obj.x_max), int(obj.y_max)), (int(obj.x_min), int(obj.y_min)),
                                        color=color, thickness=thickness)

        if legend >= LEGEND_TEXT:
            img_ann = annotate_text(obj.class_name + ' - ' + str(confidence), img_ann,
                                    (int(obj.x_min), int(obj.y_max + 5)),
                                    color)
    return img_ann


def annotate_labels(img: Image, labels: [ImgLabel], colors=None, legend=LEGEND_POSITION, thickness=None):
    img_ann = img.copy()
    for i, l in enumerate(labels):
        if colors is not None:
            color = colors[i]
        else:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0),
                      (255, 255, 255), (0, 0, 0)]
            color = colors[i % len(colors)]

        img_ann = annotate_label(img_ann, l, color, legend, thickness)

    return img_ann


def show(img: Image, name: str = "Img", t=0, labels=None, colors=None, legend=LEGEND_POSITION, thickness=None):
    img_ann = img.copy()

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels]
        img_ann = annotate_labels(img_ann, labels, colors, legend, thickness)

    imshow(img_ann, name, t)


def save_labeled(img: Image, filename: str = "Img", labels=None, colors=None, legend=LEGEND_POSITION):
    img_ann = img.copy()
    if isinstance(labels, list):
        img_ann = annotate_labels(img_ann, labels, colors, legend)
    elif isinstance(labels, ImgLabel):
        img_ann = annotate_label(img_ann, labels, colors, legend)

    imwrite(img_ann, filename + '.jpg')
