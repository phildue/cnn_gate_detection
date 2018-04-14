import numpy as np

from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.shotgen.positiongen.PositionGen import PositionGen
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class AirSimGen:

    def __init__(self, posegen: PositionGen, airsim: AirSimClient, empty_frac=0.1):
        self.empty_frac = empty_frac
        self.airsim = airsim
        self.pose_gen = posegen

    def generate(self, n_samples) -> [(Image, ImgLabel)]:
        samples = []
        labels = []
        n_empty = 0
        n_empty_max = int(np.round(self.empty_frac * n_samples))
        self.airsim.reset()
        while len(samples) < n_samples:
            pose = self.pose_gen.gen_pos()
            self.airsim.set_pose(pose)
            img, label = self.airsim.retrieve_samples()

            if len(label.objects) is 0:
                n_empty += 1
                if n_empty >= n_empty_max:
                    continue

            samples.append(img)
            labels.append(label)

        return samples, labels
