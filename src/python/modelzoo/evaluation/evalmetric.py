from modelzoo.evaluation import evaluate_file
from modelzoo.evaluation.MetricDetection import MetricDetection
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.utils import create_dirs, save_file, load_file


def evalmetric(
        name,
        batch_size,
        model_src,
        label_file,
        img_res=None,
        min_box_area=0,
        iou_thresh=0.4,
        color_format=None,
        show=False):
    # Model
    conf_thresh = 0
    summary = load_file(model_src + '/summary.pkl')
    architecture = summary['architecture']
    anchors = summary['anchors']

    if color_format is None:
        color_format = summary['color_format']

    if img_res is None:
        img_res = summary['img_res']

    model = GateNet.create_by_arch(norm=img_res,
                                   architecture=architecture,
                                   anchors=anchors,
                                   batch_size=batch_size,
                                   color_format=color_format,
                                   conf_thresh=conf_thresh,
                                   augmenter=None,
                                   weight_file=model_src + '/model.h5'
                                   )

    # Result Paths
    result_path = model_src + '/test/'
    result_file = name + '_result_metric.pkl'
    exp_param_file = name + '_evalmetric'

    create_dirs([result_path])

    exp_params = {'name': name,
                  'model': model_src,
                  'conf_thresh': conf_thresh,
                  'iou_thresh': iou_thresh,
                  'label_src': label_file,
                  'min_box_area': min_box_area}

    save_file(exp_params, exp_param_file + '.txt', result_path)
    save_file(exp_params, exp_param_file + '.pkl', result_path)

    evaluate_file(model, label_file,
                  metrics=[MetricDetection(iou_thresh=iou_thresh, show_=show,min_box_area=min_box_area*img_res[0]*img_res[1])],
                  verbose=True,
                  out_file_metric=result_path + result_file)
