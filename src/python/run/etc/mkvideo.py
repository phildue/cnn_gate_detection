from modelzoo.backend.visuals.video.videomaker import make_video
from utils.workdir import cd_work

cd_work()
make_video('logs/yolov2_aligned_distort/iros/', output='logs/yolov2_aligned_distort/iros/video.avi')
