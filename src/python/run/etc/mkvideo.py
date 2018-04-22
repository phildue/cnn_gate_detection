from modelzoo.backend.visuals.video.videomaker import make_video
from utils.workdir import cd_work

cd_work()
make_video('logs/gatev5_mixed/demo/eth/', output='logs/gatev5_mixed/demo/eth.avi')
