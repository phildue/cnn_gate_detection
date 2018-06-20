import argparse
import pprint as pp

from modelzoo.backend.tensor.CropGridLoss import CropGridLoss
from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.cropnet.CropNet2L import CropNet2L
from modelzoo.models.cropnet.CropNet import CropNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.transform.RandomBrightness import RandomBrightness
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomShift import RandomShift
from utils.imageprocessing.transform.TransformFlip import TransformFlip
from utils.workdir import cd_work

ARCHITECTURE = [{'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                {'name': 'max_pool', 'size': (2, 2)},
                {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
                {'name': 'max_pool', 'size': (2, 2)},
                {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 2, 'strides': (1, 1), 'alpha': 0.1}]
EPOCHS = 100
INITIAL_EPOCH = 0
WORK_DIR = 'test'
N_SAMPLES = None
LEARNING_RATE = 0.001
BATCH_SIZE = 4
IMAGE_SOURCE = ["resource/ext/samples/daylight/", "resource/ext/samples/industrial_new/"]
LOSS = CropGridLoss()


def train(architecture=ARCHITECTURE,
          work_dir=WORK_DIR,
          loss=LOSS,
          batch_size=BATCH_SIZE,
          n_samples=N_SAMPLES,
          epochs=EPOCHS,
          initial_epoch=INITIAL_EPOCH,
          learning_rate=LEARNING_RATE,
          image_source=IMAGE_SOURCE,
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
    augmenter = RandomEnsemble([(1.0, RandomBrightness(0.5, 2.0)),
                                (0.5, TransformFlip()),
                                (0.2, RandomShift(-.3, .3))])

    predictor = CropNet(net=CropNet2L(architecture=architecture, input_shape=(52, 52), loss=loss),
                        augmenter=augmenter)

    """
    Datasets
    """

    train_gen = GateGenerator(image_source, batch_size=batch_size, valid_frac=0.05,
                              color_format='bgr', label_format='xml', n_samples=n_samples)

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

    predictor.compile(params=params, metrics=['accuracy'])

    """
    Training Config
    """

    training = Training(predictor, train_gen,
                        out_file='model.h5',
                        patience_early_stop=20,
                        patience_lr_reduce=10,
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
                        type=str, default=ARCHITECTURE)
    parser.add_argument("--work_dir", help="Working directory", type=str, default=WORK_DIR)
    parser.add_argument("--image_source", help="List of folders to be scanned for train images",
                        default=IMAGE_SOURCE)
    parser.add_argument("--batch_size", help="Batch Size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--initial_epoch", type=int, default=INITIAL_EPOCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)

    args = parser.parse_args()

    train(architecture=args.model_name,
          work_dir=args.work_dir,
          batch_size=args.batch_size,
          n_samples=args.n_samples,
          initial_epoch=args.initial_epoch,
          learning_rate=args.learning_rate,
          epochs=args.epochs,
          image_source=args.image_source)
