# coding=utf-8
# First step
#       Group and Look at some examples
# H1:    Receptive field is too small/ too little convolutions when all features are together:
#       How to show? network with larger kernels should work better on larger boxes
#       Possible solution making it deeper/ predictors at larger scales can have smaller grid dilated convolutions at final layer to increase receptive field
# H2:    Skewed  training distribution:
#       How to show? Increase training set/ upweigh large gates and see performance
#       Solution is the same
# H3:    Context
#        Remove pole and see performance
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Imageprocessing import show, COLOR_RED, COLOR_GREEN
from utils.labels.utils import resize_label
from utils.workdir import cd_work

iou_thresh = 0.4
min_box_areas = [0.001, 0.025, 0.05, 0.1, 0.15, 0.25]

cd_work()
for min_box_area in [0.25]:
    file = load_file(
        'out/1807/baseline416x416-13x13+9layers/test/range_iou{}-area{}_result_metric.pkl'.format(
            iou_thresh,
            min_box_area))
    results = [ResultByConfidence(r) for r in file['results']['MetricDetection']]
    labels_pred = file['labels_pred']
    labels_true = file['labels_true']
    image_files = file['image_files']
    for result, label_pred, image_file, label_true in zip(results, labels_pred, image_files, labels_true):
        for i, c in enumerate(result.confidences[1:-1], 1):
            if result.results[c].false_negatives > 0 or result.results[c].false_positives > 0:
                img = imread(image_file[30:], 'bgr')
                label_p_res = resize_label(label_pred[c], (416, 416), img.shape[:2])
                label_t_res = resize_label(label_true, (416, 416), img.shape[:2])
                show(img, labels=[label_p_res, label_t_res], colors=[COLOR_RED, COLOR_GREEN])
