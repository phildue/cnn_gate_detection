from frontend.utils.BoundingBox import BoundingBox

from src.python.modelzoo.backend.tensor import non_max_suppression
from src.python.modelzoo.models.Decoder import Decoder


class Postprocessor:
    def __init__(self,
                 decoder: Decoder,
                 conf_thresh=0.3,
                 iou_thresh=0.4,
                 ):
        self.decoder = decoder
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def postprocess(self, netout):
        boxes = self.decoder.decode_netout_to_boxes_batch(netout)
        boxes = [self.non_max_suppression(b, self.iou_thresh) for b in boxes]
        boxes = [self.filter(b, self.conf_thresh) for b in boxes]
        return boxes

    @staticmethod
    def filter(predictions, conf_thresh):
        return [box for box in predictions if box.c >= conf_thresh]

    @staticmethod
    def non_max_suppression(boxes: [BoundingBox], iou_thresh):
        return non_max_suppression(boxes, iou_thresh, n_max=100)
