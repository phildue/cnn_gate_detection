from modelzoo.backend.visuals.video.videomaker import make_video
from utils.workdir import work_dir

work_dir()
make_video('logs/yolov2_aligned_distort/iros/', output='logs/yolov2_aligned_distort/iros/video.avi')
