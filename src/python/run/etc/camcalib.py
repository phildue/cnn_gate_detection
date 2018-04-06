import glob
from pprint import pprint

import numpy as np
from utils.imageprocessing.Backend import imread, resize
from utils.imageprocessing.CamCalibration import CamCalibration
from utils.workdir import cd_work

cd_work()
src_images = 'resource/camera_calibration/unreal/'
out_file = 'resource/cam_params_unreal.pkl'

files = glob.glob(src_images + '*.jpg')
images = [imread(f, 'bgr') for f in files]
cam_calib = CamCalibration(images[0].shape[:2], 'chess', (6, 6))
error = cam_calib.calibrate(images)
print("Estimation error:", error)
cam_calib.save(out_file)
cam_calib = CamCalibration.load(out_file)
pprint("Camera Matrix:")
pprint(cam_calib.camera_mat)
pprint("Distortion:")
pprint(cam_calib.distortion)
pprint("Rvecs:")
pprint(cam_calib.rotation_vectors)
pprint("Tvecs:")
pprint(cam_calib.translation_vectors)
cam_calib.demo(images[0])
