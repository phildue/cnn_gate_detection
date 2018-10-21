from utils.SetAnalysis import SetAnalysis
from utils.workdir import cd_work

cd_work()
s = SetAnalysis((416,416),'resource/ext/samples/iros2018_course_final_simple_17gates/')
s.show_summary()

