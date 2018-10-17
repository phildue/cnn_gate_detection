from modelzoo.iou import non_max_suppression
from modelzoo.models.Decoder import Decoder

from utils.labels.ObjectLabel import ObjectLabel


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
        labels = []
        for i in range(netout.shape[0]):
            label = self.decoder.decode_netout(netout[i])
            objs = self.non_max_suppression(label.objects, self.iou_thresh)
            objs = self.filter(objs, self.conf_thresh)
            label.objects = objs
            labels.append(label)
        if len(labels) > 1:
            return labels
        else:
            return labels[0]

    @staticmethod
    def filter(predictions: [ObjectLabel], conf_thresh):
        return [box for box in predictions if box.confidence >= conf_thresh]

    @staticmethod
    def non_max_suppression(boxes: [ObjectLabel], iou_thresh):
        return non_max_suppression(boxes, iou_thresh, n_max=100)
