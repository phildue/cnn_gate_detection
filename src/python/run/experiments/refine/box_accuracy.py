from run.evaluation.evalset import evalset

models = ['refnet52x52-3x3+4layers+64filters',
          'refnet52x52-3x3+6layers+16filters',
          'refnet52x52-3x3+6layers+32filters',
          'refnet52x52-3x3+6layers+64filters']
grids = [
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),

]
img_res = [
    (52, 52),
    (52, 52),
    (52, 52),
    (52, 52),
    (52, 52),
    (52, 52),
]
ious = [0.4, 0.6, 0.8]

for i, model in enumerate(models):
    for iou in ious:
        evalset(name='',
                batch_size=8,
                model_src='out/2606/' + model,
                image_source=['resource/ext/samples/industrial_new_test/', 'resource/ext/samples/daylight_test/'],
                grid=grids[i],
                img_res=img_res[i],
                iou_thresh=iou,
                n_samples=None)
