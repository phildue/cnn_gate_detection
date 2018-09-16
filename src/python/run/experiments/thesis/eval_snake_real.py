import glob

from modelzoo.evaluation import ConfidenceEvaluator
from modelzoo.evaluation.MetricDetection import MetricDetection
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.workdir import cd_work

cd_work()
results_root = 'out/thesis/snake/'
image_root = 'resource/ext/samples/'
datasets = [  # 'jevois_cyberzoo',
    # 'jevois_basement',
    'jevois_mavlab',
    'iros2018_course_final_simple_17gates']
prediction_folders = [  # 'predictions_cyberzoo',
    # 'predictions_basement',
    'predictions_mavlab',
    'predictions_iros_simplified']
box_sizes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0]
img_area = 640 * 480
for i, d in enumerate(datasets):
    parser = DatasetParser.get_parser(image_root + d + "/", 'xml', 'bgr')

    prediction_files = list(sorted(glob.glob(results_root + "test/" + prediction_folders[i] + '/*.xml')))
    predictions = [parser.read_label(f) for f in prediction_files]

    images, truth = parser.read(len(prediction_files))

    image_files = list(sorted(glob.glob(image_root + d + "/*.jpg")))

    for iou in [0.4, 0.6, 0.8]:
        box_min = box_sizes[0]
        box_max = box_sizes[-1]
        metric = MetricDetection(False, iou_thresh=iou, min_box_area=box_min * img_area,
                                 max_box_area=img_area * box_max, store=False)
        ConfidenceEvaluator(None, metrics=[metric],
                            out_file='out/thesis/snake/{}_boxes{}-{}_iou{}.pkl'.format(d, box_min, box_max,
                                                                                       iou)).evaluate(
            labels_true=truth, labels_raw=predictions, image_files=image_files)
        for j in range(len(box_sizes) - 1):
            box_min = box_sizes[j]
            box_max = box_sizes[j + 1]
            metric = MetricDetection(False, iou_thresh=iou, min_box_area=box_min * img_area,
                                     max_box_area=img_area * box_max, store=False)
            ConfidenceEvaluator(None, metrics=[metric],
                                out_file='out/thesis/snake/{}_boxes{}-{}_iou{}.pkl'.format(d, box_min, box_max,
                                                                                           iou)).evaluate(
                labels_true=truth, labels_raw=predictions, image_files=image_files)
