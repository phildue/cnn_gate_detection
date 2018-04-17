import numpy as np

from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.shotgen.positiongen.PositionGen import PositionGen
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class AirSimGen:

    def __init__(self, posegen: PositionGen, airsim: AirSimClient, empty_frac=0.1, range_magnitude=(4, 30),
                 max_angle=30, org_aspect_ratio=1.05):
        self.org_aspect_ratio = org_aspect_ratio
        self.max_angle = max_angle
        self.range_magnitude = range_magnitude
        self.empty_frac = empty_frac
        self.airsim = airsim
        self.pose_gen = posegen

    def generate(self, n_samples) -> [(Image, ImgLabel)]:
        samples = []
        labels = []
        n_empty = 0
        n_empty_max = int(np.round(self.empty_frac * n_samples))
        max_aspect_ratio = self.org_aspect_ratio / (self.max_angle / 90)
        # self.airsim.reset()
        while len(samples) < n_samples:
            pose = self.pose_gen.gen_pos()
            self.airsim.set_pose(pose)
            img, label = self.airsim.retrieve_samples()

            if len(label.objects) is 0:
                n_empty += 1
                if n_empty < n_empty_max:
                    samples.append(img)
                    labels.append(label)
            else:
                bad_sample = False
                for o in label.objects:
                    bad_sample = not (self.range_magnitude[0] < o.pose.magnitude < self.range_magnitude[1] and
                                      o.width > 0 and
                                      o.height / o.width < max_aspect_ratio)
                    if bad_sample: break

                if not bad_sample:
                    samples.append(img)
                    labels.append(label)

        return samples, labels
