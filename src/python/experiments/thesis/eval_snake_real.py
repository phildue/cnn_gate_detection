import glob

from modelzoo.evaluation.MetricDetection import DetectionEvaluator

from modelzoo.evaluation import ConfidenceEvaluator
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.workdir import cd_work

cd_work()
results_root = 'out/thesis/snake/'
image_root = 'resource/ext/samples/'
datasets = [
    # 'jevois_cyberzoo',
    'jevois_basement',
    # 'jevois_hallway',
    # 'iros2018_course_final_simple_17gates'
    # 'real_test_labeled'
]
prediction_folders = [
    # 'predictions_cyberzoo',
    'predictions_basement',
    # 'predictions_hallway',
    # 'predictions_simu'
    # 'predictions_all'
]
box_sizes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
img_area = 640 * 480

for i, d in enumerate(datasets):
    parser = DatasetParser.get_parser(image_root + d + "/", 'xml', 'bgr')

    images, truth = parser.read()
    image_files = list(sorted(glob.glob(image_root + d + "/*.jpg")))

    for iou in [0.4, 0.6, 0.8]:
        for n in range(5):
            prediction_files = list(
                sorted(glob.glob(results_root + "test/" + prediction_folders[i] + "/" + str(n) + '/*.xml')))
            predictions = [parser.read_label(f) for f in prediction_files]

            box_min = box_sizes[0]
            box_max = box_sizes[-1]
            metric = DetectionEvaluator(False, iou_thresh=iou, min_box_area=box_min * img_area,
                                        max_box_area=img_area * box_max, store=False)
            ConfidenceEvaluator(None, metrics=[metric],
                                out_file='out/thesis/snake/{}_boxes{}-{}_iou{}_i{}.pkl'.format(d, box_min,
                                                                                               box_max,
                                                                                               iou,
                                                                                               n)).evaluate(
                labels_true=truth, labels_raw=predictions, image_files=image_files)
            for j in range(len(box_sizes) - 1):
                box_min = box_sizes[j]
                box_max = box_sizes[j + 1]
                metric = DetectionEvaluator(False, iou_thresh=iou, min_box_area=box_min * img_area,
                                            max_box_area=img_area * box_max, store=False)
                ConfidenceEvaluator(None, metrics=[metric],
                                    out_file='out/thesis/snake/{}_boxes{}-{}_iou{}_i{}.pkl'.format(d, box_min, box_max,
                                                                                                   iou, n)).evaluate(
                    labels_true=truth, labels_raw=predictions, image_files=image_files)
