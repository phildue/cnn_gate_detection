import argparse
import os

from utils.imageprocessing.Backend import imwrite
from utils.imageprocessing.RTPViewer import RtpViewer
from utils.workdir import cd_work

cd_work()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int, default=5000,
                    help="The port number to open for the RTP stream (5000 or 6000)")
parser.add_argument("-s", "--scale", type=float, default=1.,
                    help="The scaling factor to apply to the incoming video stream (default: 1)")
parser.add_argument("-r", "--rotate", type=int, default=0,
                    help="The number of clockwise 90deg rotations to apply to the stream [0-3] (default: 0)")

args = parser.parse_args()

filename = os.path.dirname(os.path.abspath(__file__)) + "/rtp_" + str(args.port) + ".sdp"

viewer = RtpViewer(filename)
viewer.scale = args.scale
viewer.rotate = args.rotate

if not viewer.cap.isOpened():
    viewer.cleanup()
    raise IOError("Can't open video stream")

viewer.run()
viewer.cleanup()

for i, frame in enumerate(viewer.recorded_frames):
    imwrite(frame, 'resource/calibration_images/{0:03d}.jpg'.format(i))
