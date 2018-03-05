from backend.visuals.video.videomaker import make_video
from workdir import work_dir

work_dir()
make_video('logs/tinyyolo_10k/stream3/images/', output='logs/tinyyolo_10k/stream3/video.avi')
