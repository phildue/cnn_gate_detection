from samplegen.labelmaker.LabelMaker import LabelMaker
from utils.fileaccess.utils import create_dirs
from utils.workdir import cd_work

cd_work()

img_dir = 'resource/samples/video/'
src_file = 'resource/video/eth.avi'
set_dir = 'resource/samples/video/eth/'

create_dirs([img_dir, set_dir])

LabelMaker(None, img_dir, step_size=10, set_dir=set_dir, start_idx=8).parse()
