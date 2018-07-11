from modelzoo.backend.visuals.plots.BaseMultiPlot import BaseMultiPlot
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()

legends = []
mean_recalls = []
mean_precisions = []
total_recall = []
total_precision = []
linestyle = ['-.', '-*', '-x', '-o', '--']
for model in [
    'out/1807/gatenet-strided416x416-13x13+9layers+pyramid',
    'out/1807/gatenet416x416-13x13+9layers+pyramid',
    'out/1807/graygatenet416x416-13x13+9layers+pyramid',
    'out/1807/mobilegatenet416x416-13x13+9layers+pyramid',
    'out/1807/wr_basic_gatenet416x416-13x13+10layers+pyramid',
    # 'out/1807/wr_inception_gatenet416x416-13x13+10layers+pyramid'
]:
    results = load_file(model + '/test/test_result_metric.pkl')
    detections = [ResultByConfidence(r) for r in results['results']['MetricDetection']]
    mean_pr, mean_recall = average_precision_recall(detections)
    total_results = sum_results(detections)

    legends.append(model)
    mean_recalls.append(mean_recall)
    mean_precisions.append(mean_pr)
    total_recall.append(total_results.recalls)
    total_precision.append(total_results.precisions)

pr_img = BaseMultiPlot(x_data=mean_recalls,
                       y_data=mean_precisions,
                       y_label='Precision',
                       x_label='Recall',
                       y_lim=(0, 1.0),
                       legend=legends,
                       title='Precision Recall Per Image',
                       line_style=linestyle,
                       x_res=None)

pr_total = BaseMultiPlot(x_data=total_recall,
                         y_data=total_precision,
                         y_label='Precision',
                         x_label='Recall',
                         y_lim=(0, 1.0),
                         legend=legends,
                         title='Precision Recall Total',
                         line_style=linestyle,
                         x_res=None)

pr_img.show(False)
pr_total.show()
