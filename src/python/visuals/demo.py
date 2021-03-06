

from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import show, LEGEND_TEXT, save_labeled, LEGEND_POSITION
from utils.labels.ImgLabel import ImgLabel


def demo_generator(model: Predictor, generator: DatasetGenerator, iou_thresh=0.4, t_show=-1, out_file=None,
                   n_samples=None, size=None):
    if n_samples is None:
        n_samples = generator.n_samples
    iterator = iter(generator.generate_valid())
    idx = 0
    for i in range(int(n_samples / generator.batch_size)):
        batch = next(iterator)
        for i in range(len(batch)):

            img = batch[i][0]
            label = batch[i][1]
            # show(img, 'demo',
            #      colors=[(255, 255, 255), (0, 0, 255), (255, 0, 0)],
            #      legend=LEGEND_TEXT, t=t_show)
            label_pred = model.predict(img)
            if img.shape[0] != model.input_shape[0] or img.shape[1] != model.input_shape[1]:
                img, label = model.preprocessor.crop_to_input(img, label)
                img, label = resize(img, model.input_shape, label=label)
            #            print(BoundingBox.from_label(label_pred))

            boxes_pred =label_pred.objects
            boxes_true = label.objects

            false_negatives = []
            false_positives = boxes_pred.copy()
            true_positives = []
            for j in range(len(boxes_true)):
                match = False
                box_true = boxes_true[j]
                for k in range(len(false_positives)):
                    box_pred = false_positives[k]
                    match = box_pred.poly.iou(box_true.poly) > iou_thresh and box_pred.class_id == box_true.class_id
                    if match:
                        true_positives.append(box_pred)
                        false_positives.remove(box_pred)
                        break
                if not match:
                    false_negatives.append(box_true)

            label_tp = ImgLabel(true_positives)
            label_fp = ImgLabel(false_positives)
            label_fn = ImgLabel(false_negatives)
            show(img, 'demo', labels=[label_tp, label_fp, label],
                 colors=[(255, 255, 255), (0, 0, 255), (255, 0, 0)],
                 legend=LEGEND_POSITION, t=t_show)

            if out_file is not None:
                save_labeled(img.bgr, out_file + '/{0:04d}.jpg'.format(idx), labels=[label_tp, label_fp, label],
                             colors=[(255, 255, 255), (0, 0, 255), (255, 0, 0)],
                             legend=LEGEND_TEXT)
                idx += 1
