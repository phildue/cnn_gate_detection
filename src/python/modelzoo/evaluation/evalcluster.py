import numpy as np

from modelzoo.evaluation.DetectionEvaluator import DetectionEvaluator
from utils.SetAnalysis import SetAnalysis
from utils.labels.ImgLabel import ImgLabel


def evalcluster_yaw_dist(labels_true, labels_pred, conf_thresh, n_bins_angle=10, n_bins_dist=10, iou_thresh=0.6):
    angles = np.linspace(0, 180, n_bins_angle)
    distances = np.linspace(0, 12, n_bins_dist)
    tp = np.zeros((n_bins_dist, n_bins_angle))
    fn = np.zeros((n_bins_dist, n_bins_angle))

    for i in range(len(labels_true)):
        m = DetectionEvaluator(show_=True, min_box_area=0.0 * 416 * 416, max_box_area=2.0 * 416 * 416,
                               min_aspect_ratio=0,
                               max_aspect_ratio=100.0, iou_thresh=iou_thresh)
        label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
        label_true = labels_true[i]
        m.evaluate(label_true, label_pred)

        for b in m.boxes_fn:
            yaw = np.degrees(b.pose.yaw)
            if yaw < 0:
                yaw *= -1
            if yaw > 180:
                yaw = 360 - yaw
            yaw = 180 - yaw
            dist = np.linalg.norm(b.pose.transvec / 3)
            i, j = SetAnalysis.assign_angle_dist_to_bin(yaw, dist, angles, distances)
            fn[i, j] += 1
        for i_b, b in enumerate(m.boxes_true):
            if i_b not in m.false_negatives_idx:
                yaw = np.degrees(b.pose.yaw)
                if yaw < 0:
                    yaw *= -1
                if yaw > 180:
                    yaw = 360 - yaw
                yaw = 180 - yaw
                dist = np.linalg.norm(b.pose.transvec / 3)
                i, j = SetAnalysis.assign_angle_dist_to_bin(yaw, dist, angles, distances)
                tp[i, j] += 1

    return fn, tp


def evalcluster_height_width(labels_true, labels_pred, conf_thresh, n_bins, iou_thresh=0.6):
    widths = np.linspace(0, 416, n_bins)
    heights = np.linspace(0, 416, n_bins)
    tp = np.zeros((n_bins, n_bins))
    fp = np.zeros((n_bins, n_bins))
    fn = np.zeros((n_bins, n_bins))
    evaluator = DetectionEvaluator(show_=True, min_box_area=0.0 * 416 * 416, max_box_area=2.0 * 416 * 416,
                                   min_aspect_ratio=0,
                                   max_aspect_ratio=100.0, iou_thresh=iou_thresh)
    for i in range(len(labels_true)):

        label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
        label_true = labels_true[i]
        evaluator.evaluate(label_true, label_pred)

        for b in evaluator.boxes_fn:
            w = b.poly.width
            h = b.poly.height
            for i_w in range(len(widths) - 1):
                if widths[i_w] <= w < widths[i_w + 1]:
                    break
            for i_h in range(len(heights) - 1):
                if heights[i_h] <= h < heights[i_h + 1]:
                    break

            fn[i_h, i_h] += 1

        for b in evaluator.boxes_fp:
            w = b.poly.width
            h = b.poly.height
            for i_w in range(len(widths) - 1):
                if widths[i_w] <= w < widths[i_w + 1]:
                    break
            for i_h in range(len(heights) - 1):
                if heights[i_h] <= h < heights[i_h + 1]:
                    break

            fp[i_h, i_h] += 1

        for b in evaluator.boxes_tp:
            w = b.poly.width
            h = b.poly.height
            for i_w in range(len(widths) - 1):
                if widths[i_w] <= w < widths[i_w + 1]:
                    break
            for i_h in range(len(heights) - 1):
                if heights[i_h] <= h < heights[i_h + 1]:
                    break

            tp[i_h, i_h] += 1

    return tp, fp, fn


def evalscatter_wh(labels_true, labels_pred, conf_thresh, iou_thresh=0.6):
    evaluator = DetectionEvaluator(show_=True, min_box_area=0.01 * 416 * 416, max_box_area=1.0 * 416 * 416,
                                   min_aspect_ratio=0,
                                   max_aspect_ratio=100.0, iou_thresh=iou_thresh)
    fn = []
    fp = []
    tp = []
    for i in range(len(labels_true)):

        label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
        label_true = labels_true[i]
        evaluator.evaluate(label_true, label_pred)

        for b in evaluator.boxes_fn:
            w = b.poly.width
            h = b.poly.height
            fn.append((w,h))

        for b in evaluator.boxes_fp:
            w = b.poly.width
            h = b.poly.height
            fp.append((w,h))

        for i_b, b in enumerate(evaluator.boxes_true):
            if i_b not in evaluator.false_negatives_idx:
                w = b.poly.width
                h = b.poly.height
                tp.append((w,h))

    return tp, fp, fn