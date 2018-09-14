from utils import BoundingBox


def match(box_true, boxes_pred, iou_thresh):
    matches = []
    for i, b in enumerate(boxes_pred):
        if box_true.iou(b) > iou_thresh and \
                box_true.prediction == b.prediction:
            matches.append(i)

    return matches


def detections(boxes_true: [BoundingBox], boxes_pred: [BoundingBox], iou_thresh):
    true_positives = []
    false_negatives = []
    for b in boxes_true:
        box_matches = match(b, boxes_pred, iou_thresh)
        if len(box_matches) is 0:
            false_negatives.append(b)
        else:
            new_matches = [m for m in box_matches if m not in true_positives]
            if len(new_matches) > 0:
                true_positives.append(new_matches[0])

    return len(true_positives), len(false_negatives), len(boxes_pred) - len(true_positives)
