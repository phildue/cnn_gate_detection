import numpy as np

from modelzoo.evaluation.DetectionEvaluator import DetectionEvaluator
from modelzoo.evaluation.utils import sum_results, average_precision_recall
from utils.SetAnalysis import SetAnalysis
from utils.labels.ImgLabel import ImgLabel


def evalcluster_yaw_dist(labels_true, labels_pred, conf_thresh, n_bins_angle=10, n_bins_dist=10, iou_thresh=0.6):
    angles = np.linspace(0, 180, n_bins_angle)
    distances = np.linspace(0, 12, n_bins_dist)
    tp = np.zeros((n_bins_dist, n_bins_angle))
    fn = np.zeros((n_bins_dist, n_bins_angle))

    for i in range(len(labels_true)):
        m = DetectionEvaluator(min_box_area=0.01 * 416 * 416, max_box_area=2.0 * 416 * 416,
                               min_aspect_ratio=0.3,
                               max_aspect_ratio=3.0, iou_thresh=iou_thresh)
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
    evaluator = DetectionEvaluator(min_box_area=0.0 * 416 * 416, max_box_area=2.0 * 416 * 416,
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

            fn[i_w, i_h] += 1

        for b in evaluator.boxes_fp:
            w = b.poly.width
            h = b.poly.height
            for i_w in range(len(widths) - 1):
                if widths[i_w] <= w < widths[i_w + 1]:
                    break
            for i_h in range(len(heights) - 1):
                if heights[i_h] <= h < heights[i_h + 1]:
                    break

            fp[i_w, i_h] += 1

        for b in evaluator.boxes_tp:
            w = b.poly.width
            h = b.poly.height
            for i_w in range(len(widths) - 1):
                if widths[i_w] <= w < widths[i_w + 1]:
                    break
            for i_h in range(len(heights) - 1):
                if heights[i_h] <= h < heights[i_h + 1]:
                    break

            tp[i_w, i_h] += 1

    return tp, fp, fn


def evalcluster_size(labels_true, labels_pred, conf_thresh, n_bins, iou_thresh=0.6, min_size=0, max_size=416):
    size = np.linspace(min_size, max_size, n_bins)
    tp = np.zeros((n_bins,))
    fp = np.zeros((n_bins,))
    fn = np.zeros((n_bins,))
    evaluator = DetectionEvaluator(min_box_area=0.0 * 416 * 416, max_box_area=2.0 * 416 * 416,
                                   min_aspect_ratio=0,
                                   max_aspect_ratio=100.0, iou_thresh=iou_thresh)
    for i in range(len(labels_true)):

        label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
        label_true = labels_true[i]
        evaluator.evaluate(label_true, label_pred)

        for b in evaluator.boxes_fn:
            a = b.poly.area
            matched = False
            for i_w in range(len(size) - 1):
                if a < size[0]:
                    print('Area too small')
                    break

                if size[i_w] <= a < size[i_w + 1]:
                    matched = True
                    break
            if matched:
                fn[i_w] += 1
            else:
                print('FN Not matched: {}'.format(b))
                print(a)

        for b in evaluator.boxes_fp:
            a = b.poly.area
            matched = False
            for i_w in range(len(size) - 1):
                if a < size[0]:
                    print('Area too small')
                    break
                if size[i_w] <= a < size[i_w + 1]:
                    matched = True
                    break
            if matched:
                fp[i_w] += 1
            else:
                print('FP Not matched: {}'.format(b))
                print(a)
        for b in evaluator.boxes_tp:
            a = b.poly.area
            matched = False
            for i_w in range(len(size) - 1):
                if a < size[0]:
                    print('Area too small')
                    break
                if size[i_w] <= a < size[i_w + 1]:
                    matched = True
                    break
            if matched:
                tp[i_w] += 1
            else:
                print('TP Not matched: {}'.format(b))
                print(a)
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
            fn.append((w, h))

        for b in evaluator.boxes_fp:
            w = b.poly.width
            h = b.poly.height
            fp.append((w, h))

        for i_b, b in enumerate(evaluator.boxes_true):
            if i_b not in evaluator.false_negatives_idx:
                w = b.poly.width
                h = b.poly.height
                tp.append((w, h))

    return tp, fp, fn


def evalset(labels_true, labels_pred, iou_thresh=0.6):
    evaluator = DetectionEvaluator(min_box_area=0.01 * 416 * 416, max_box_area=1.0 * 416 * 416,
                                   min_aspect_ratio=0.3,
                                   max_aspect_ratio=3.0, iou_thresh=iou_thresh)
    tp = []
    fp = []
    fn = []
    boxes_true = []
    results = []
    for i in range(len(labels_true)):
        evaluator.evaluate(labels_true[i], labels_pred[i])
        tp.extend(evaluator.boxes_tp)
        fp.extend(evaluator.boxes_fp)
        fn.extend(evaluator.boxes_fn)
        boxes_true.extend(evaluator.boxes_true)
        results.append(evaluator.result)
    sum_r = sum_results(results)
    return sum_r,tp, fp, fn, boxes_true


def evalcluster_size_ap(labels_true, labels_pred, n_bins, iou_thresh=0.6, min_size=0, max_size=2.0, img_res=(416, 416)):
    size = np.linspace(min_size * img_res[0] * img_res[1], max_size * img_res[0] * img_res[1], n_bins + 1)

    evaluator = DetectionEvaluator(min_box_area=min_size * img_res[0] * img_res[1],
                                   max_box_area=max_size * img_res[0] * img_res[1],
                                   min_aspect_ratio=0,
                                   max_aspect_ratio=100.0, iou_thresh=iou_thresh)

    aps = []
    n_true = []
    for i_s in range(len(size) - 1):
        results_bin = []
        n_true_bin = 0
        for i, l_true in enumerate(labels_true):
            l_true_bin = ImgLabel([o for o in l_true.objects if size[i_s] < o.poly.area < size[i_s + 1]])
            l_pred_bin = ImgLabel([o for o in labels_pred[i].objects if size[i_s] < o.poly.area < size[i_s + 1]])
            result = evaluator.evaluate(l_true_bin, l_pred_bin)
            results_bin.append(result)
            n_true_bin += len(l_true_bin.objects)
        r = sum_results(results_bin)
        mean_pr, mean_rec, std_pr, std_rec = average_precision_recall([r])
        ap = np.mean(mean_pr)
        aps.append(ap)
        n_true.append(n_true_bin)

    return aps, n_true
