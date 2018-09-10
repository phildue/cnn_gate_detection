from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()

legends = []
mean_recalls = []
mean_precisions = []
total_recalls = []
total_precisions = []
linestyle = ['-.', '-*', '-x', '-o', '--']

models = [
    'baseline104x104-13x13+9layers',
    # 'baseline208x208-13x13+9layers',
    'baseline416x416-13x13+9layers',
    # 'baseline52x52-13x13+9layers',
    'bottleneck416x416-13x13+9layers',
    # 'bottleneck_narrow416x416-13x13+9layers',
    # 'bottleneck_narrow_strides416x416-13x13+9layers',
    # 'combined208x208-13x13+13layers',
    'grayscale416x416-13x13+9layers',
    # 'narrow416x416-13x13+9layers',
    # 'narrow_strides416x416-13x13+9layers',
    'narrow_strides_late_bottleneck416x416-13x13+9layers',
    #    'strides2416x416-13x13+9layers',
]

names = [
    'baseline104x104',
    'baseline416x416',
    'bottleneck416x416',
    'grayscale416x416',
    'bottleneck_large_strides416x416',
]
iou = 0.6
for model in models:
    results = load_file('out/1807/' + model + '/test/total_iou{}-area0.0_result_metric.pkl'.format(iou))
    detections = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
    meanAP, mean_recall = average_precision_recall(detections)
    total_results = sum_results(detections)

    legends.append(model)
    mean_recalls.append(mean_recall)
    mean_precisions.append(meanAP)

    total_recalls.append(total_results.recalls[1:])
    total_precisions.append(total_results.precisions[1:])



pr_img.show(False)
pr_total.show()
