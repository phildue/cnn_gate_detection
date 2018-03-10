import glob

from utils.imageprocessing.Backend import imread
from utils.imageprocessing.CamCalibration import CamCalibration
from utils.workdir import work_dir

work_dir()
src_images = 'resource/camera_calibration/'
out_file = 'resource/cam_params.csv'

files = glob.glob(src_images + '*.jpg')
images = [imread(f, 'bgr') for f in files]

cam_calib = CamCalibration(images[0].shape[:2], 'chess')
error = cam_calib.calibrate(images)

print("Estimation error:", error)
if error < 0.1:
    cam_calib.save(out_file)
    print("Parameters stored at: ", out_file)
    cam_calib.demo(images[0])