from modelzoo.evaluation import evaluate_generator
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file, load_file


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
        result_file=None):

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
    generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg', n_samples=n_samples,
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

    evaluate_generator(model, generator, verbose=True, out_file_labels=result_path + result_file)
