from modelzoo.evaluation.DetectionEvaluator import DetectionEvaluator
from utils.fileaccess.utils import create_dirs, save_file, load_file
from utils.imageprocessing.Backend import imread


def evalmetric(
        name,
        model_src,
        label_file,
        img_res=None,
        min_box_area=0,
        max_box_area=2.0,
        iou_thresh=0.4,
        show=False,
        result_path=None,
        result_file=None,
        min_aspect_ratio=0.0,
        max_aspect_ratio=100.0):
    # Model
    conf_thresh = 0
    summary = load_file(model_src + '/summary.pkl')

    if img_res is None:
        img_res = summary['img_res']

    exp_param_file = name + '_evalmetric'

    exp_params = {'name': name,
                  'model': model_src,
                  'conf_thresh': conf_thresh,
                  'iou_thresh': iou_thresh,
                  'label_src': label_file,
                  'min_box_area': min_box_area,
                  'max_box_area': max_box_area}

    metric = DetectionEvaluator(
        iou_thresh=iou_thresh,
        min_box_area=min_box_area * img_res[0] * img_res[1],
        max_box_area=max_box_area * img_res[0] * img_res[1],
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )

    content = load_file(label_file)
    labels_true = content['labels_true']
    labels_pred = content['labels_pred']
    image_files = content['image_files']
    results = []
    for i in range(len(labels_true)):
        result = metric.evaluate(labels_true[i], labels_pred[i])
        if show:
            metric.show(imread(image_files[i],'bgr'))
        results.append(result)

    output = {
        'results': results,
        'labels_true': labels_true,
        'labels_pred': labels_pred,
        'image_files': image_files
    }

    if result_path:
        create_dirs([result_path])
        if result_file is None:
            result_file = name
        save_file(output, result_file, result_path)
        save_file(exp_params, exp_param_file + '.txt', result_path)
        save_file(exp_params, exp_param_file + '.pkl', result_path)

    return output
