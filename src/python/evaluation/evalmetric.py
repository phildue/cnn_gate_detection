from evaluation.DetectionEvaluator import DetectionEvaluator
from utils.ModelSummary import ModelSummary
from utils.fileaccess.utils import create_dirs, save_file, load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.transform.ImgTransform import ImgTransform


def evaluate_labels(
        name,
        model_src,
        label_file,
        preprocessing: [ImgTransform] = None,
        min_box_area=0,
        max_box_area=2.0,
        iou_thresh=0.4,
        show=-1,
        result_path=None,
        result_file=None,
        min_aspect_ratio=0.0,
        max_aspect_ratio=100.0):
    exp_param_file = name + '_summary'

    exp_params = {'name': name,
                  'model': model_src,
                  'iou_thresh': iou_thresh,
                  'label_src': label_file,
                  'min_box_area': min_box_area,
                  'max_box_area': max_box_area}

    summary = ModelSummary.from_file(model_src+'/summary.pkl')

    metric = DetectionEvaluator(
        iou_thresh=iou_thresh,
        min_box_area=min_box_area*summary.img_size,
        max_box_area=max_box_area*summary.img_size,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )

    content = load_file(label_file)
    labels_true = content['labels_true']
    labels_pred = content['labels_pred']
    image_files = content['image_files']
    results = []
    for i in range(len(labels_true)):
        img = imread(image_files[i], 'bgr')
        label = labels_true[i]
        if preprocessing:
            for p in preprocessing:
                img, label = p.transform(img, label)
        result = metric.evaluate(label, labels_pred[i])
        if show > -1:
            metric.show(img, t=show)
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
