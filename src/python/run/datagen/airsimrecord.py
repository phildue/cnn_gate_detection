# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode


from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.airsim.AirSimRecord import AirSimRecord
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.fileaccess.utils import create_dirs
from utils.timing import tic, toc
from utils.workdir import cd_work

cd_work()
name = "daylight_flight"
shot_path = "resource/ext/samples/" + name + "/"

n_samples = 100

#TODO choose simulation environment here + camera settings and start simulation

client = AirSimClient()
samplegen = AirSimRecord(client, 0.2)

create_dirs([shot_path])
set_writer = DatasetParser.get_parser(shot_path,
                                      image_format='jpg',
                                      label_format='xml',
                                      color_format='bgr',
                                      start_idx=0)

client.reset()
tic()
samples, labels = samplegen.generate(n_samples=n_samples)
set_writer.write(samples, labels)
toc("{} samples generated after ".format(len(samples)))
