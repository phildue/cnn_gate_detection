import glob
import numpy as np
from utils.imageprocessing.Backend import imread, resize
from utils.imageprocessing.CamCalibration import CamCalibration
from utils.workdir import cd_work

cd_work()
src_images = 'resource/camera_calibration/'
out_file = 'resource/cam_params_bebop.pkl'

files = glob.glob(src_images + '*.jpg')
images = [imread(f, 'bgr') for f in files]
# cam_calib = CamCalibration(images[0].shape[:2], 'chess', (4, 5))
# error = cam_calib.calibrate(images)
#
# print("Estimation error:", error)
# cam_calib.save(out_file)
cam_calib = CamCalibration.load(out_file)
cam_calib.demo(images[0])
