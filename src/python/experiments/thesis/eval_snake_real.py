import glob

from evaluation.DetectionEvaluator import DetectionEvaluator
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.utils import save_file
from utils.workdir import cd_work

cd_work()
results_root = 'out/thesis/snake/'
image_root = 'resource/ext/samples/'
datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
    # 'iros2018_course_final_simple_17gates'
    # 'real_test_labeled'
]
prediction_folders = [
    'predictions_cyberzoo',
    'predictions_basement',
    'predictions_hallway',
    # 'predictions_simu'
    # 'predictions_all'
]
img_area = 640 * 480

for i, d in enumerate(datasets):
    parser = DatasetParser.get_parser(image_root + d + "/", 'xml', 'bgr')

    images, truth = parser.read()
    image_files = list(sorted(glob.glob(image_root + d + "/*.jpg")))

    for iou in [0.4, 0.6, 0.8]:
        eval = DetectionEvaluator(min_box_area=0 * img_area,
                                  max_box_area=img_area * 4.0, store=False, min_aspect_ratio=0, max_aspect_ratio=100.0,
                                  iou_thresh=iou)
        for n in range(5):
            prediction_files = list(
                sorted(glob.glob(results_root + "test/" + prediction_folders[i] + "/" + str(n) + '/*.xml')))
            predictions = [parser.read_label(f) for f in prediction_files]

            results = []
            for j in range(len(predictions)):
                result = eval.evaluate(truth[j], predictions[j])
                results.append(result)

            output = {
                'results': results,
                'labels_true': truth,
                'labels_pred': predictions,
                'image_files': image_files
            }

            save_file(output, 'test_' + d + '_results_iou' + str(iou) + '_' + str(n) + '.pkl', results_root)
