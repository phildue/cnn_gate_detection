import argparse
import pprint as pp

import numpy as np

from modelzoo.backend.tensor.Training import Training
from modelzoo.backend.tensor.yolo.AveragePrecisionYolo import AveragePrecisionYolo
from modelzoo.models.yolo.Yolo import Yolo
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

BATCH_SIZE = 16
N_SAMPLES = None
EPOCHS = 100
INITIAL_EPOCH = 0
LEARNING_RATE = 0.001
IMG_HEIGHT = 416
IMG_WIDTH = 416


def train(architecture,
          work_dir,
          image_source,
          anchors,
          augmenter,
          min_obj_size,
          max_obj_size,
          max_aspect_ratio,
          min_aspect_ratio,
          img_res=(IMG_HEIGHT, IMG_WIDTH),
          batch_size=BATCH_SIZE,
          n_samples=N_SAMPLES,
          initial_epoch=INITIAL_EPOCH,
          learning_rate=LEARNING_RATE,
          epochs=EPOCHS,
          weight_file=None,
          validation_set=None,
          class_names='gate'):

    def learning_rate_schedule(epoch):
        if epoch > 50:
            return 0.0001
        else:
            return 0.001

    def filter(label):
        objs_in_size = [obj for obj in label.objects if
                        min_obj_size < (obj.height * obj.width) / (img_res[0] * img_res[1]) < max_obj_size]

        objs_within_angle = [obj for obj in objs_in_size if
                             min_aspect_ratio < obj.height / obj.width < max_aspect_ratio]

        objs_in_view = []
        for obj in objs_within_angle:
            mat = np.array([[obj.x_min, obj.y_max],
                            [obj.x_max, obj.y_max]])
            if (len(mat[(mat[:, 0] < 0) | (mat[:, 0] > img_res[1])]) +
                len(mat[(mat[:, 1] < 0) | (mat[:, 1] > img_res[0])])) > 2:
                continue
            objs_in_view.append(obj)

        return ImgLabel(objs_in_size)

    cd_work()

    if weight_file is None and initial_epoch > 0:
        weight_file = 'out/' + work_dir + '/model.h5'

    """
    Model
    """

    predictor = Yolo.create_by_arch(class_names=class_names,
                                    architecture=architecture,
                                    color_format='bgr',
                                    anchors=anchors,
                                    batch_size=batch_size,
                                    augmenter=augmenter,
                                    norm=img_res,
                                    weight_file=weight_file,
                                    conf_thresh=0.5,
                                    iou_thresh=0.4)

    """
    Datasets
    """

    valid_frac = 0.05
    valid_gen = None
    if validation_set is not None:
        valid_gen = GateGenerator(validation_set, batch_size=batch_size, valid_frac=1.0,
                                  color_format='bgr', label_format='xml', filter=filter, max_empty=0).generate_valid()
        valid_frac = 0.0

    train_gen = GateGenerator(image_source, batch_size=batch_size, valid_frac=valid_frac,
                              color_format='bgr', label_format='xml', n_samples=n_samples,
                              remove_filtered=filter, max_empty=0.0)

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

    def average_precision06(y_true, y_pred):
        return AveragePrecisionYolo(batch_size=batch_size, n_boxes=predictor.n_boxes, grid=predictor.grid,
                                    norm=predictor.norm, iou_thresh=0.6, iou_thresh_nms=0.4,
                                    n_classes=len(class_names)).compute(y_true, y_pred)

    predictor.compile(params=params, metrics=[average_precision06])

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
                        validation_generator=valid_gen
                        )

    summary = training.summary
    summary['architecture'] = architecture
    summary['anchors'] = anchors
    summary['img_res'] = img_res
    summary['grid'] = predictor.grid
    summary['valid_set'] = validation_set
    summary['min_obj_size'] = min_obj_size
    summary['max_obj_size'] = max_obj_size
    summary['min_aspect_ratio'] = min_aspect_ratio
    summary['max_aspect_ratio'] = max_aspect_ratio
    pp.pprint(summary)
    save_file(summary, 'summary.txt', result_path, verbose=False)
    save_file(summary, 'summary.pkl', result_path, verbose=False)
    predictor.net.backend.summary()

    training.fit_generator()



