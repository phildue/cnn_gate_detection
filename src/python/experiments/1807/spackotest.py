from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Imageprocessing import show, COLOR_GREEN, COLOR_RED
from utils.labels.utils import resize_label
from utils.workdir import cd_work

models = [
    # 'baseline104x104-13x13+9layers',
    'baseline208x208-13x13+9layers',
    'baseline416x416-13x13+9layers',
    'baseline52x52-13x13+9layers',
    'bottleneck416x416-13x13+9layers',
    # 'bottleneck_narrow416x416-13x13+9layers',
    # 'bottleneck_narrow_strides416x416-13x13+9layers',
    # 'combined208x208-13x13+13layers',
    # 'grayscale416x416-13x13+9layers',
    # 'narrow416x416-13x13+9layers',
    # 'narrow_strides416x416-13x13+9layers',
    # 'narrow_strides_late_bottleneck416x416-13x13+9layers',
    #    'strides2416x416-13x13+9layers',
]
cd_work()
for model in models:
    results = load_file('out/1807/' + model + '/test/range_iou0.4-area0.001_result_metric.pkl')
    for i, r in enumerate(results['results']['MetricDetection']):
        for j, key in enumerate(r.keys()):
            dr = r[key]
            if dr.true_positives < 0 or dr.false_positives < 0 or dr.false_negatives < 0:
                print("Warning")
                print(dr)
                print(key)
                labels_pred = results['labels_pred'][i][key]
                labels_true = results['labels_true'][i]
                image_file = results['image_files'][i][30:]
                print(image_file)
                img = imread(image_file, 'bgr')
                labels_pred = resize_label(labels_pred, (104, 104), img.shape[:2])
                labels_true = resize_label(labels_true, (104, 104), img.shape[:2])
                show(img, labels=[labels_pred, labels_true],colors=[COLOR_RED,COLOR_GREEN])
