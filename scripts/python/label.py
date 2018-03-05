import os

from labelmaker.LabelMaker import LabelMaker
from workdir import work_dir

work_dir()

out_dir = 'resource/samples/video/'
src_file = 'resource/videos/eth.avi'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

LabelMaker(src_file, out_dir, step_size=10).parse()
