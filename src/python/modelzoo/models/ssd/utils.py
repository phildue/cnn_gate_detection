from imageprocessing.Backend import resize
from imageprocessing.Imageprocessing import show, LEGEND_TEXT, COLOR_GREEN

from src.python.utils import BoundingBox
from src.python.utils.imageprocessing.Image import Image


def show_encoding(label_t, img: Image, shape=(600, 600)):
    background = label_t[label_t[:, 0] == 1]
    foreground = label_t[label_t[:, 0] != 1]
    boxes_fg = BoundingBox.from_tensor_centroid(foreground[:, 1:-4], foreground[:, -4:])
    boxes_bg = BoundingBox.from_tensor_centroid(background[:, :-4], background[:, -4:])

    label_bg = BoundingBox.to_label(boxes_bg)
    label_fg = BoundingBox.to_label(boxes_fg)

    img_large, label_bg = resize(img, shape, label=label_bg)
    _, label_fg = resize(img, shape, label=label_fg)

    show(img_large, labels=[label_fg, label_bg],
         colors=[COLOR_GREEN, (100, 100, 100)],
         thickness=2,
         name='Anchors',
         legend=LEGEND_TEXT)
