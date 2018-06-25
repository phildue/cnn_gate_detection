import argparse
import pprint as pp

import numpy as np

from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.gatenet.AveragePrecisionGateNet import AveragePrecisionGateNet
from modelzoo.models.ModelFactory import ModelFactory
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.CropGenerator import CropGenerator
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.workdir import cd_work

MODEL_NAME = [{'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                    {'name': 'max_pool', 'size': (2, 2)},
                    {'name': 'conv_leaky', 'kernel_size': (6, 6), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1}]
WORK_DIR = 'test'
BATCH_SIZE = 4
N_SAMPLES = 500
EPOCHS = 100
INITIAL_EPOCH = 0
LEARNING_RATE = 0.001
IMAGE_SOURCE = ["resource/ext/samples/daylight/", "resource/ext/samples/industrial_new/"]
IMG_HEIGHT = 52
IMG_WIDTH = 52
ANCHORS = np.array([[[1, 1],
                     [0.3, 0.3],
                     [2, 1],
                     [1, 0.5],
                     [0.7, 0.7]
                     ]])
AUGMENTER = RandomEnsemble([(1.0, RandomBrightness(0.5, 2.0)),
                            (0.5, TransformFlip()),
                            (0.2, RandomShift(-.3, .3))])


def train(architecture=MODEL_NAME,
          work_dir=WORK_DIR,
          batch_size=BATCH_SIZE,
          n_samples=N_SAMPLES,
          initial_epoch=INITIAL_EPOCH,
          learning_rate=LEARNING_RATE,
          epochs=EPOCHS,
          image_source=IMAGE_SOURCE,
          img_res=(IMG_HEIGHT, IMG_WIDTH),
          anchors=ANCHORS,
          augmenter=AUGMENTER
          ):
    def learning_rate_schedule(epoch):
        if epoch > 50:
            return 0.0001
        else:
            return 0.001

    cd_work()

    """
    Model
    """

    if isinstance(architecture, str):
        predictor = ModelFactory.build(architecture, batch_size, src_dir=None, img_res=img_res, grid=[(13, 13)])
    else:
        predictor = GateNet.create_by_arch(norm=img_res,architecture=architecture, anchors=anchors, batch_size=batch_size, augmenter=augmenter)

    """
    Datasets
    """

    train_gen = CropGenerator(GateGenerator(image_source, batch_size=batch_size, valid_frac=0.05,
                              color_format='bgr', label_format='xml', n_samples=n_samples))

    """
    Paths
    """
    result_path = 'out/' + work_dir + '/'

    create_dirs([result_path, result_path + '/results/'])

    """
    Optimizer Config
    """
    params = {'optimizer': 'adam',
              'lr': learning_rate,
              'beta_1': 0.9,
              'beta_2': 0.999,
              'epsilon': 1e-08,
              'decay': 0.0005}

    def average_precision(y_true, y_pred):
        return AveragePrecisionGateNet(batch_size=batch_size, n_boxes=predictor.n_boxes, grid=predictor.grid,
                                       norm=predictor.norm).compute(y_true, y_pred)

    predictor.compile(params=params, metrics=[average_precision])

    """
    Training Config
    """

    training = Training(predictor, train_gen,
                        out_file='model.h5',
                        patience_early_stop=5,
                        patience_lr_reduce=3,
                        log_dir=result_path,
                        stop_on_nan=True,
                        lr_schedule=learning_rate_schedule,
                        initial_epoch=initial_epoch,
                        epochs=epochs,
                        log_csv=True,
                        lr_reduce=0.1,
                        )

    summary = training.summary
    summary['architecture'] = architecture
    pp.pprint(summary)
    save_file(summary, 'summary.txt', result_path, verbose=False)
    save_file(summary, 'summary.pkl', result_path, verbose=False)
    predictor.net.backend.summary()

    training.fit_generator()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model name",
                        type=str, default=MODEL_NAME)
    parser.add_argument("--work_dir", help="Working directory", type=str, default=WORK_DIR)
    parser.add_argument("--image_source", help="List of folders to be scanned for train images",
                        default=IMAGE_SOURCE)
    parser.add_argument("--batch_size", help="Batch Size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--initial_epoch", type=int, default=INITIAL_EPOCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--img_width", default=IMG_WIDTH, type=int)
    parser.add_argument("--img_height", default=IMG_HEIGHT, type=int)

    args = parser.parse_args()

    train(
        architecture=args.model_name,
        work_dir=args.work_dir,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        initial_epoch=args.initial_epoch,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        image_source=args.image_source,
        img_res=(args.img_height, args.img_width)
    )
