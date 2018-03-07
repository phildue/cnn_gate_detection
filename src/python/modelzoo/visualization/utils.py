import numpy as np

from modelzoo.models.Predictor import Predictor
from utils.BoundingBox import BoundingBox
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.imageprocessing.Imageprocessing import show, LEGEND_TEXT, save_labeled
from utils.labels.utils import resize_label
from utils.imageprocessing.Backend import resize


def demo_generator(model: Predictor, generator: DatasetGenerator, iou_thresh=0.4, t_show=-1, out_file=None):
    iterator = iter(generator.generate_valid())
    idx = 0
    while True:
        batch = next(iterator)
        for i in range(len(batch)):

            img = batch[i][0]
            label = batch[i][1]
            label_pred = model.predict(img)
            label_pred = resize_label(label_pred, model.input_shape, img.shape[:2])
            boxes_pred = BoundingBox.from_label(label_pred)
            boxes_true = BoundingBox.from_label(label)

            boxes_correct_idx = []
            for b_true in boxes_true:
                for j, b_pred in enumerate(boxes_pred):
                    if b_pred.iou(b_true) > iou_thresh and b_pred.prediction == b_true.prediction:
                        boxes_correct_idx.append(int(j))
                        break

            boxes_correct = np.where(np.arange(0, len(boxes_pred)) == boxes_correct_idx, boxes_pred, None)
            boxes_correct = [b for b in boxes_correct if b is not None]

            boxes_wrong = np.where(np.arange(0, len(boxes_pred)) != boxes_correct_idx, boxes_pred, None)
            boxes_wrong = [b for b in boxes_wrong if b is not None]
            label_correct = BoundingBox.to_label(boxes_correct)
            label_wrong = BoundingBox.to_label(boxes_wrong)
            show(img.bgr, 'demo', labels=[label, label_correct, label_wrong],
                 colors=[(0, 255, 0), (255, 255, 255), (0, 0, 255)],
                 legend=LEGEND_TEXT, t=t_show)

            if out_file is not None:
                save_labeled(img.bgr, out_file + '/{0:04d}.jpg'.format(idx), labels=[label_pred],
                             colors=[(255, 0, 0)],
                             legend=LEGEND_TEXT)
                idx += 1
