import glob
import random

from imageprocessing.Backend import imread

from src.python.samplegen.shotgen import ShotGen
from src.python.utils.fileaccess import LabelFileParser


class ShotLoad(ShotGen):
    def __init__(self, shot_path: str, img_format='jpg', random=True):
        self.random = random
        self.format = img_format
        self.path = shot_path

    def get_shots(self, n_shots=10):
        shots = []
        labels = []
        files = list(sorted(glob.glob(self.path + "*." + self.format)))
        if n_shots > len(files):
            print("Not enough shots available on hard drive, getting: " + str(len(files)))

        for i, f in enumerate(files):
            if i >= n_shots: break
            if self.random:
                f = random.choice(files)

            shots.append(imread(f))
            labels.append(LabelFileParser.read_label_file_pkl(f.replace(self.format, "pkl")))
        return shots, labels
