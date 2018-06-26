import numpy as np

from modelzoo.backend.tensor.cropnet.CropNet2L import CropNet2L
from modelzoo.evaluation.DetectionResult import DetectionResult
from modelzoo.evaluation.ResultsByConfidence import ResultByConfidence
from modelzoo.models.cropnet.CropNet import CropNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file, load_file
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work


def evalset(
        model_src='out/cropnet416x416->3x3+3layers+32filters/',
        name='',
        batch_size=8,
        img_res=(416, 416),
        grid=(3, 3),
        image_source=['resource/ext/samples/industrial_new_test/'],
        n_samples=100
):
    cd_work()

    # Image Source
    color_format = 'bgr'

    # Model
    summary = load_file(model_src + '/summary.pkl')
    architecture = summary['architecture']
    model = CropNet(input_shape=img_res, output_shape=grid,
                    color_format='yuv',
                    net=CropNet2L(architecture=architecture,
                                  input_shape=img_res,
                                  weight_file=model_src + '/model.h5'
                                  ))

    # Evaluator

    # Result Paths
    result_path = model_src + '/results/'
    result_file = 'result_' + name + '.pkl'
    result_img_path = result_path + 'images_' + name + '/'
    exp_param_file = 'eval_' + name + '.txt'

    create_dirs([result_path, result_img_path])
    generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format='jpg', n_samples=n_samples,
                              shuffle=False, color_format=color_format, label_format='xml', start_idx=0)
    gen = generator.generate()
    results = []
    for i in range(0, generator.n_samples, batch_size):
        batch = next(gen)
        for j in range(batch_size):
            if i + j >= generator.n_samples: break
            x, y_true = model.preprocessor.preprocess_test([batch[j]])
            y_true = y_true.astype(np.float32)
            y_pred = model.net.predict(x)

            result = {}
            for k, conf_thresh in enumerate(np.linspace(0, 1.0, 11, dtype=np.float32)):
                tp = y_pred[(y_pred == y_true) & (y_true > conf_thresh)].size
                fp = y_pred[(y_pred != y_true) & (y_pred > conf_thresh)].size
                fn = y_pred[(y_pred != y_true) & (y_true > conf_thresh)].size
                tn = y_true[(y_pred == y_true) & (y_true < conf_thresh)].size
                result[conf_thresh] = DetectionResult(true_positives=tp,
                                                      false_negatives=fn,
                                                      false_positives=fp,
                                                      true_negatives=tn)
            results.append(ResultByConfidence(result))
            # label_true = Image(y_true[0], 'bgr')
            # label_pred = Image(y_pred[0], 'bgr')
            # img = Image(x[0], 'bgr')
            # print(precision[i+j])
            # print(recall[i+j])
    #            show(label_true, name='true', t=1)
    #            show(label_pred, name='pred', t=1)
    #            show(img, name='img')

    exp_params = {'name': name,
                  'model': model.net.__class__.__name__,
                  'image_source': image_source,
                  'color_format': color_format,
                  'n_samples': generator.n_samples}

    save_file(exp_params, exp_param_file, result_path)
    save_file(results, result_path + result_file)


if __name__ == '__main__':
    evalset()
