from workdir import work_dir

from backend.visuals.videomaker import make_video

work_dir()
make_video('logs/tinyyolo_10k/stream3/images/', output='logs/tinyyolo_10k/stream3/video.avi')
