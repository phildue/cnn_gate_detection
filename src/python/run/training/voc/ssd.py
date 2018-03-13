import pprint as pp

from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.ssd.AveragePrecisionSSD import AveragePrecisionSSD
from modelzoo.models.ssd.SSD import SSD
from utils.fileaccess.VocGenerator import VocGenerator
from utils.fileaccess.utils import save_file, create_dirs
from utils.imageprocessing.transform.YoloAugmenter import YoloImgTransform
from utils.workdir import work_dir

work_dir()

batch_size = 32

image_source = 'voc'
work_path = 'logs/ssd300_voc/'

predictor = SSD.ssd300(n_classes=20, batch_size=batch_size, alpha=.1, weight_file=work_path + '/SSD300.h5')
data_generator = VocGenerator(batch_size=batch_size, shuffle=True, n_samples=64)

augmenter = YoloImgTransform()

model_name = predictor.net.__class__.__name__

epochs = 120
loss = predictor.loss


def average_precision(y_true, y_pred):
    return AveragePrecisionSSD(batch_size=batch_size, n_boxes=8096).compute(y_true, y_pred)


predictor.compile(params=None, metrics=[average_precision]
                  )
predictor.preprocessor.augmenter = augmenter


def lr_schedule(epoch):
    if 0 <= epoch < 80:
        return 0.001
    elif 80 <= epoch <= 100:
        return 0.0001
    else:
        return 0.000001


training = Training(predictor, data_generator,
                    out_file=model_name + '.h5',
                    patience=-1,
                    log_dir=work_path,
                    stop_on_nan=True,
                    initial_epoch=0,
                    epochs=epochs,
                    log_csv=True,
                    lr_reduce=0.1,
                    lr_schedule=lr_schedule)

create_dirs([work_path])

pp.pprint(training.summary)

save_file(training.summary, 'summary.txt', work_path, verbose=False)

training.fit_generator()
