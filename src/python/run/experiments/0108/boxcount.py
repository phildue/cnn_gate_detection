from samplegen.setanalysis.SetAnalyzer import SetAnalyzer
from utils.workdir import cd_work

cd_work()
SetAnalyzer((416, 416), ['resource/ext/samples/daylight_flight/',
                         'resource/ext/samples/industrial_flight/']).area_distribution_hist().show()
