from samplegen.labelmaker.LabelMaker import LabelMaker
from utils.fileaccess.utils import create_dir
from utils.workdir import work_dir

work_dir()

img_dir = 'resource/samples/video/'
src_file = 'resource/videos/eth.avi'
set_dir = 'resource/samples/video/eth/'

create_dir([img_dir, set_dir])

LabelMaker(None, img_dir, step_size=10, set_dir=set_dir, start_idx=150).parse()
