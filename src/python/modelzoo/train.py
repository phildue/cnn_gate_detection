import pprint as pp

from modelzoo.Training import Training
from modelzoo.models.gatenet.AveragePrecisionGateNet import AveragePrecisionGateNet
from modelzoo.models.gatenet.GateNet import GateNet
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

N_SAMPLES = None
EPOCHS = 100
INITIAL_EPOCH = 0
LEARNING_RATE = 0.001


def train(architecture,
          work_dir,
          batch_size,
          image_source,
          anchors,
          img_res,
          augmenter,
          color_format,
          input_channels=3,
          n_samples=N_SAMPLES,
          initial_epoch=INITIAL_EPOCH,
          learning_rate=LEARNING_RATE,
          epochs=EPOCHS,
          weight_file=None,
          n_polygon=4,
          min_aspect_ratio=0,
          max_aspect_ratio=100.0,
          max_obj_size=5,
          min_obj_size=0,
          validation_set=None,
          subsets:[float]=None
          ):
    def learning_rate_schedule(epoch):
        if epoch > 50:
            return 0.0001
        else:
            return 0.001

    cd_work()

    if weight_file is None and initial_epoch > 0:
        weight_file = 'out/' + work_dir + '/model.h5'

    """
    Model
    """

    predictor = GateNet.create_by_arch(architecture, anchors=anchors, batch_size=batch_size, augmenter=augmenter,
                                       norm=img_res, input_channels=input_channels, weight_file=weight_file,
                                       n_polygon=n_polygon, color_format=color_format)

    """
    Datasets
    """

    def filter(label):

        objs_in_size = [obj for obj in label.objects if
                        min_obj_size < (obj.poly.height * obj.poly.width) / (img_res[0] * img_res[1]) < max_obj_size]

        objs_within_angle = [obj for obj in objs_in_size if
                             min_aspect_ratio < obj.poly.height / obj.poly.width < max_aspect_ratio]

        objs_in_view = []
        for obj in objs_within_angle:
            mat = obj.poly.points
            if (len(mat[(mat[:, 0] < 0) | (mat[:, 0] > img_res[1])]) +
                len(mat[(mat[:, 1] < 0) | (mat[:, 1] > img_res[0])])) > 2:
                continue
            objs_in_view.append(obj)

        return ImgLabel(objs_in_view)

    valid_frac = 0.05
    valid_gen = None
    if validation_set is not None:
        valid_gen = GateGenerator(validation_set, batch_size=batch_size, valid_frac=1.0,
                                  color_format='bgr', label_format='xml', filter=filter, max_empty=0).generate_valid()
        valid_frac = 0.0

    train_gen = GateGenerator(image_source, batch_size=batch_size, valid_frac=valid_frac,
                              color_format='bgr', label_format='xml', n_samples=n_samples,
                              remove_filtered=False, max_empty=0, filter=filter,subsets=subsets)

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

    if n_polygon == 4:
        def average_precision06(y_true, y_pred):
            return AveragePrecisionGateNet(batch_size=batch_size, n_boxes=predictor.n_boxes, grid=predictor.grid,
                                           norm=predictor.norm, iou_thresh=0.6).compute(y_true, y_pred)

        predictor.compile(params=params, metrics=[average_precision06])
    else:
        predictor.compile(params=params)

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
    summary['max_aspect_ratio'] = max_aspect_ratio
    summary['min_aspect_ratio'] = min_aspect_ratio
    pp.pprint(summary)
    save_file(summary, 'summary.txt', result_path, verbose=False)
    save_file(summary, 'summary.pkl', result_path, verbose=False)
    predictor.net.backend.summary()

    training.fit_generator()
