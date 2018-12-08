import pprint as pp
from pathlib import Path

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, TerminateOnNaN, ReduceLROnPlateau, CSVLogger, \
    History
from keras.optimizers import Adam

from modelzoo.Decoder import Decoder
from modelzoo.Encoder import Encoder
from modelzoo.Preprocessor import Preprocessor
from modelzoo.build_model import build_detector
from modelzoo.metrics.GateDetectionLoss import GateDetectionLoss
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import create_dirs, save_file
from utils.imageprocessing.transform.RandomBlur import RandomBlur
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.RandomMotionBlur import RandomMotionBlur
from utils.imageprocessing.transform.TransformResize import TransformResize
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

cd_work()
img_res_data = 416, 416
img_res_network = 320, 320
for i in range(1, 2):
    model_dir = '320_i{0:02d}'.format(i)
    initial_epoch = 0
    epochs = 100

    architecture = [
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 4, 'strides': (1, 1), 'alpha': 0.1},
        # {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 8, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'max_pool', 'size': (2, 2)},
        {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'predict'},
        {'name': 'route', 'index': [10]},
        {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'upsample', 'size': 2},
        {'name': 'crop', 'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
        {'name': 'route', 'index': [-1, 9]},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'predict'},
        {'name': 'route', 'index': [10]},
        {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'upsample', 'size': 4},
        {'name': 'crop', 'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
        {'name': 'route', 'index': [-1, 7]},
        {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
        {'name': 'predict'}
    ]

    work_dir = 'out/ext/' + model_dir + '/'
    create_dirs([work_dir])
    """
    Datasets
    """
    image_source = ['resource/ext/samples/daylight_course1',
                    'resource/ext/samples/daylight_course5',
                    'resource/ext/samples/daylight_course3',
                    'resource/ext/samples/iros2018_course1',
                    'resource/ext/samples/iros2018_course5',
                    'resource/ext/samples/iros2018_flights',
                    'resource/ext/samples/basement_course3',
                    'resource/ext/samples/basement_course1',
                    'resource/ext/samples/iros2018_course3_test',
                    'resource/ext/samples/flight_basement_mavlab_gates_bg0',
                    'resource/ext/samples/flight_basement_mavlab_gates_bg1'
                    # 'resource/ext/samples/various_environments20k',
                    # 'resource/ext/samples/realbg20k'
                    ]
    batch_size = 16
    n_samples = 30000
    subsets = None
    min_obj_size = 0.001
    max_obj_size = 2
    min_aspect_ratio = 0.3
    max_aspect_ratio = 3.0


    def filter(label):
        objs_in_size = [obj for obj in label.objects if
                        min_obj_size < (obj.poly.height * obj.poly.width) / (
                                    img_res_data[0] * img_res_data[1]) < max_obj_size]

        objs_within_angle = [obj for obj in objs_in_size if
                             min_aspect_ratio < obj.poly.height / obj.poly.width < max_aspect_ratio]

        objs_in_view = []
        for obj in objs_within_angle:
            mat = obj.poly.points
            if (len(mat[(mat[:, 0] < 0) | (mat[:, 0] > img_res_data[1])]) +
                len(mat[(mat[:, 1] < 0) | (mat[:, 1] > img_res_data[0])])) > 2:
                continue
            objs_in_view.append(obj)

        return ImgLabel(objs_in_view)


    valid_frac = 0.005
    train_gen = GateGenerator(image_source, batch_size=batch_size, valid_frac=valid_frac,
                              color_format='bgr', label_format='xml', n_samples=n_samples,
                              remove_filtered=False, max_empty=0, filter=filter, subsets=subsets)

    """
    Model
    """
    anchors = np.array([[[330.58, 333.99],
                         [226.27, 232.06]],

                        [[153.54, 158.03],
                         [97.34, 105.7]],

                        [[59.81, 66.41],
                         [24.36, 39.8]]])
    anchors /= (416 / img_res_network[0])
    print(anchors)
    augmenter = [RandomEnsemble([
        (0.2, RandomMotionBlur(1.0, 2.0, 15)),
        (0.2, RandomBlur((5, 5))),
        # (0.3, TransformDistort(
        #     dist_model=BarrelDistortion((416, 416), rad_dist_params=[0.7, 0], tangential_dist_params=[0.7, 0])))
    ])]

    model, output_grids = build_detector(img_shape=(img_res_network[0], img_res_network[1], 3),
                                         architecture=architecture,
                                         anchors=anchors,
                                         n_polygon=4)
    encoder = Encoder(anchor_dims=anchors, img_norm=img_res_network, grids=output_grids, n_polygon=4, iou_min=0.4)
    decoder = Decoder(anchor_dims=anchors, norm=img_res_network, grid=output_grids, n_polygon=4)
    preprocessor = Preprocessor(preprocessing=[TransformResize(img_res_network)], encoder=encoder, n_classes=1,
                                img_shape=img_res_network, color_format='bgr',
                                augmentation=augmenter)
    loss = GateDetectionLoss()
    model.load_weights('out/blur_distortion_i01/model.h5', skip_mismatch=True, by_name=True)

    """
    Training Config
    """
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005)
    # metric = AveragePrecisionGateNet(batch_size=batch_size, n_boxes=encoder.n_boxes, grid=output_grids,
    #                                  norm=img_res, iou_thresh=0.6)

    # def ap60(y_true, y_pred):
    #     return metric.compute(
    #         y_true, y_pred)

    model.compile(optimizer=optimizer,
                  loss=loss.compute)

    log_file_name = work_dir + '/log.csv'
    append = Path(log_file_name).is_file() and initial_epoch > 0
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min',
                      verbose=1),
        ModelCheckpoint(work_dir + '/model.h5', monitor='val_loss', verbose=1,
                        save_best_only=True,
                        mode='min', save_weights_only=False,
                        period=1),
        TensorBoard(batch_size=batch_size, log_dir=work_dir, write_images=True,
                    histogram_freq=0),

        TerminateOnNaN(),

        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_lr=0.00001),

        CSVLogger(log_file_name, append=append),

        History()

    ]
    if isinstance(preprocessor.augmenter, RandomEnsemble):
        augmentation = ''
        augmenters = preprocessor.augmenter.augmenters
        probs = preprocessor.augmenter.probs
        for i in range(len(augmenters)):
            augmentation += '\n{0:.2f} -> {1:s}'.format(probs[i], augmenters[i].__class__.__name__)
    else:
        augmentation = preprocessor.augmenter.__class__.__name__

    summary = {'resolution': img_res_network,
               'image_source': train_gen.source_dir,
               'color_format': train_gen.color_format,
               'batch_size': train_gen.batch_size,
               'n_samples': train_gen.n_samples,
               'transform': augmentation,
               'initial_epoch': initial_epoch,
               'epochs': epochs,
               'weights': model.count_params(),
               'architecture': architecture,
               'anchors': anchors,
               'grid': output_grids,
               # 'valid_set': validation_set,
               'min_obj_size': min_obj_size,
               'max_obj_size': max_obj_size,
               'max_aspect_ratio': max_aspect_ratio,
               'min_aspect_ratio': min_aspect_ratio}

    pp.pprint(summary)
    save_file(summary, 'summary.txt', work_dir, verbose=False)
    save_file(summary, 'summary.pkl', work_dir, verbose=False)
    model.summary()
    model.fit_generator(
        generator=preprocessor.preprocess_train_generator(train_gen.generate()),
        steps_per_epoch=(train_gen.n_samples / batch_size),
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        validation_data=preprocessor.preprocess_train_generator(train_gen.generate_valid()),
        validation_steps=int(train_gen.n_samples * train_gen.valid_frac) / batch_size,
        callbacks=callbacks)
