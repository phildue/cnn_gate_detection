import glob

from imageprocessing.Backend import split_video, imread
from imageprocessing.GateAnnotater import GateAnnotater
from imageprocessing.Imageprocessing import show
from labels.GateLabel import GateLabel
from labels.ImgLabel import ImgLabel
from labels.Pose import Pose

from src.python.samplegen.scene import GateCorners
from src.python.utils.fileaccess.SetFileParser import SetFileParser


class LabelMaker:
    def __init__(self, video_file, img_dir, set_dir, step_size=1, start_idx=0, label_format='pkl', img_format='jpg'):
        self.step_size = step_size
        self.img_dir = img_dir
        self.video_file = video_file
        self.set_writer = SetFileParser(set_dir, img_format, label_format, start_idx)
        self.start_idx = start_idx

    def parse(self):
        if self.video_file is not None:
            split_video(self.video_file, self.img_dir)
        files = sorted(glob.glob(self.img_dir + '/*.jpg'))

        for i in range(self.start_idx * self.step_size, len(files), self.step_size):
            img = imread(files[i], 'bgr')
            objects = GateAnnotater(img.copy()).annotate()
            print(objects)
            if len(objects) > 0:
                gate_labels = []
                for obj in objects:
                    if len(obj) >= 4:
                        bottom_left, bottom_right, top_right, top_left = obj
                        cx = int(bottom_left[0] + (top_right[0] - bottom_left[0]) / 2)
                        cy = int(bottom_left[1] + (top_right[1] - bottom_left[1]) / 2)
                        gate_label = GateLabel(position=Pose(),
                                               gate_corners=GateCorners(top_left=top_left, top_right=top_right,
                                                                        bottom_left=bottom_left,
                                                                        bottom_right=bottom_right,
                                                                        center=(cx, cy)))
                        gate_labels.append(gate_label)
                img_label = ImgLabel(gate_labels)
                show(img, labels=img_label, name='image')
                self.set_writer.write([img], [img_label])
