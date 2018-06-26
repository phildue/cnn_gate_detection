import numpy as np

from utils.workdir import cd_work


def evaluate(netname, img_res, grid, layers, filters, old=True):
    if old:
        if filters == 16:
            folder_name = '{}{}x{}-{}layers'.format(netname, img_res[0], img_res[1], layers)
        else:
            folder_name = '{}{}x{}-{}layers-{}filters'.format(netname, img_res[0], img_res[1], layers, filters)
    else:
        folder_name = '{}{}x{}->{}x{}+{}layers+{}filters'.format(netname, img_res[0], img_res[1], grid[0],
                                                                 grid[1], layers, filters)

    from run.evaluation.cropnet.evalset import evalset
    evalset(model_src='out/'+folder_name +'/',
            img_res=img_res,
            grid=grid,
            image_source=['resource/ext/samples/industrial_new_test/', 'resource/ext/samples/daylight_test/'],
            n_samples=10)

cd_work()
for l in np.array([l for l in range(2, 9)]):
    evaluate('cropnet', (52, 52), (13, 13), l, 16)
    evaluate('cropnet', (52, 52), (13, 13), l, 32)
    evaluate('cropnet', (52, 52), (13, 13), l, 48)
    evaluate('cropnet', (104, 104), (13, 13), l, 16)
#
for l in [3, 5, 7, 9, 11] :
    evaluate('cropnet', (416, 416), (3, 3), l, 64, False)
