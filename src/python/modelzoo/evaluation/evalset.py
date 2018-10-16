from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file, load_file
from utils.timing import tic, toc


def evalset(
        name,
        batch_size,
        model_src,
        img_res=None,
        image_source=['resource/ext/samples/industrial_new_test/'],
        n_samples=None,
        color_format_dataset='bgr',
        preprocessing=None,
        color_format=None,
        result_path=None,
        result_file=None,
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

    model = GateNet.create_by_arch(norm=img_res,
                                   architecture=architecture,
                                   anchors=anchors,
                                   batch_size=batch_size,
                                   color_format=color_format,
                                   conf_thresh=conf_thresh,
                                   augmenter=None,
                                   preprocessor=preprocessing,
                                   weight_file=model_src + '/model.h5'
                                   )

    # Evaluator

    # Result Paths
    if result_path is None:
        result_path = model_src + '/test/'
    if result_file is None:
        result_file = name + '_results.pkl'

    exp_param_file = name + '_evalset'

    create_dirs([result_path])
    generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format=image_format,
                              n_samples=n_samples,
                              shuffle=False, color_format=color_format_dataset, label_format='xml', start_idx=0)

    exp_params = {'name': name,
                  'model': model_src,
                  'conf_thresh': conf_thresh,
                  'image_source': image_source,
                  'color_format': color_format_dataset,
                  'n_samples': generator.n_samples,
                  'result_file': result_file,
                  'preprocessing': preprocessing}

    save_file(exp_params, exp_param_file + '.pkl', result_path)
    save_file(exp_params, exp_param_file + '.txt', result_path)
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
        predictions = model.predict(images)
        if image_files[0].shape[:2] != model.input_shape:
            print("Evaluator:: Labels have different size")

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
        save_file(content, result_file,result_path)
