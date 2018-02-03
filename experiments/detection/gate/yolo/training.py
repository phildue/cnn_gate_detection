# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from workdir import work_dir

work_dir()
from fileaccess.GateGenerator import GateGenerator
from models.Yolo.Yolo import Yolo
from fileaccess.utils import save
from backend.training import fit_generator

BATCH_SIZE = 8

yolo = Yolo(batch_size=BATCH_SIZE, class_names=['gate'])

trainset = GateGenerator(directory='resource/samples/mult_gate', batch_size=BATCH_SIZE, valid_frac=0.1)
train_params = {'optimizer': 'adam',
                'lr': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-08,
                'decay': 0.0005}

yolo.compile(train_params)

training_history = fit_generator(yolo, trainset, out_file=result_path + 'yolo-gate-adam.h5', batch_size=BATCH_SIZE,
                                 initial_epoch=0)
save(training_history, 'training_history.pkl', result_path)
