import glob
from pprint import pprint

from utils.imageprocessing.Backend import imread
from utils.imageprocessing.CamCalibration import CamCalibration
from utils.workdir import cd_work

cd_work()
src_images = 'resource/camera_calibration/jevois/'
out_file = 'resource/camera_calibration/jevois.pkl'

files = glob.glob(src_images + '*.jpg')
images = [imread(f, 'bgr') for f in files]
cam_calib = CamCalibration((images[0].shape[1], images[0].shape[0]), 'chess', (9, 6))
error = cam_calib.calibrate(images)
print("Estimation error:", error)
cam_calib.save(out_file)
cam_calib = CamCalibration.load(out_file)
pprint("Camera Matrix:")
pprint(cam_calib.camera_mat)
pprint("Distortion:")
pprint(cam_calib.distortion)
pprint("Rvecs:")
# pprint(cam_calib.rotation_vectors)
pprint("Tvecs:")
# pprint(cam_calib.translation_vectors)
cam_calib.demo(images[0])
