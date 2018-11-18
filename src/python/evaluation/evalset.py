from modelzoo.Decoder import Decoder
from modelzoo.Encoder import Encoder
from modelzoo.Postprocessor import Postprocessor
from modelzoo.Preprocessor import Preprocessor
from modelzoo.build_model import build_detector
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file, load_file
from utils.imageprocessing.Imageprocessing import show
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc


def infer_on_set(
        model_src,
        image_source,
        batch_size,
        result_path,
        result_file,
        img_res=None,
        n_samples=None,
        color_format_dataset='bgr',
        preprocessing=None,
        color_format=None,
        image_format="jpg"):
    # Model
    conf_thresh = 0
    summary = load_file(model_src + '/summary.pkl')
    architecture = summary['architecture']
    anchors = summary['anchors']
    if color_format is None:
        color_format = summary['color_format']

    if img_res is None:
        img_res = summary['img_res']

    model, output_grids = build_detector(img_shape=(img_res[0], img_res[1], 3), architecture=architecture,
                                         anchors=anchors,
                                         n_polygon=4)
    model.load_weights(model_src + '/model.h5')
    encoder = Encoder(anchor_dims=anchors, img_norm=img_res, grids=output_grids, n_polygon=4, iou_min=0.4)
    preprocessor = Preprocessor(preprocessing=preprocessing, encoder=encoder, n_classes=1, img_shape=img_res,
                                color_format=color_format)
    decoder = Decoder(anchor_dims=anchors, n_polygon=4, norm=img_res, grid=output_grids)
    postproessor = Postprocessor(decoder=decoder)

    generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format=image_format,
                              n_samples=n_samples,
                              shuffle=False, color_format=color_format_dataset, label_format='xml', start_idx=0)

    create_dirs([result_path])

    exp_params = {'model': model_src,
                  'conf_thresh': conf_thresh,
                  'image_source': image_source,
                  'color_format': color_format_dataset,
                  'n_samples': generator.n_samples,
                  'result_file': result_file,
                  'preprocessing': preprocessing}

    save_file(exp_params, 'test_summary' + '.pkl', result_path)
    save_file(exp_params, 'test_summary' + '.txt', result_path)
    n_batches = int(generator.n_samples / generator.batch_size)
    it = iter(generator.generate())
    labels_true = []
    labels_pred = []
    image_files = []
    for i in range(n_batches):
        batch = next(it)
        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        image_files_batch = [b[2] for b in batch]

        tic()
        predictions = postproessor.postprocess(model.predict(preprocessor.preprocess_batch(images)))
        if images[0].shape[0] != model.input_shape[1] or \
                images[0].shape[1] != model.input_shape[2]:
            print("Evaluator:: Labels have different size")

        for j, p in enumerate(predictions):
            l = p.copy()
            img_show = images[j].copy()
            l.objects = [o for o in l.objects if o.confidence > 0.01]
            if preprocessing:
                for p in preprocessing:
                    img_show, _ = p.transform(img_show, ImgLabel([]))
            show(img_show, labels=l, t=1)
        # labels = [resize_label(l, images[0].shape[:2], self.model.input_shape) for l in labels]
        # for j in range(len(batch)):
        #     show(batch[j][0], labels=[predictions[j], labels[j]])

        labels_true.extend(labels)
        labels_pred.extend(predictions)
        image_files.extend(image_files_batch)
        toc("Evaluated batch {0:d}/{1:d} in ".format(i, n_batches))

        content = {'labels_true': labels_true,
                   'labels_pred': labels_pred,
                   'image_files': image_files}
        save_file(content, result_file, result_path)
