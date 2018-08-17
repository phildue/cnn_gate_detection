from modelzoo.backend.visuals.video.videomaker import make_video
from utils.workdir import cd_work

cd_work()
make_video('resource/New folder/', output='resource/daylight.avi')
