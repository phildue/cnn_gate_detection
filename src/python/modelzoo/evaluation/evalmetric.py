from modelzoo.evaluation import evaluate_file
from modelzoo.evaluation.MetricDetection import MetricDetection
from utils.fileaccess.utils import create_dirs, save_file, load_file


def evalmetric(
        name,
        model_src,
        label_file,
        img_res=None,
        min_box_area=0,
        max_box_area=1.0,
        iou_thresh=0.4,
        show=False,
        result_path=None,
        result_file=None,
        min_aspect_ratio=None,
        max_aspect_ratio=None):
    # Model
    conf_thresh = 0
    summary = load_file(model_src + '/summary.pkl')


    if img_res is None:
        img_res = summary['img_res']

    # Result Paths
    if result_path is None:
        result_path = model_src + '/test/'

    if result_file is None:
        result_file = name + '.pkl'

    exp_param_file = name + '_evalmetric'

    create_dirs([result_path])

    exp_params = {'name': name,
                  'model': model_src,
                  'conf_thresh': conf_thresh,
                  'iou_thresh': iou_thresh,
                  'label_src': label_file,
                  'min_box_area': min_box_area,
                  'max_box_area': max_box_area}

    save_file(exp_params, exp_param_file + '.txt', result_path)
    save_file(exp_params, exp_param_file + '.pkl', result_path)

    evaluate_file(label_file,
                  metrics=[MetricDetection(iou_thresh=iou_thresh, show_=show,
                                           min_aspect_ratio=min_aspect_ratio,
                                           max_aspect_ratio=max_aspect_ratio,
                                           min_box_area=min_box_area * img_res[0] * img_res[1],
                                           max_box_area=max_box_area * img_res[0] * img_res[1])],
                  verbose=True,
                  out_file_metric=result_path + result_file)
