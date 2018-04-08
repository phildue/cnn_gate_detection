from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.shotgen.positiongen.PositionGen import PositionGen
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class AirSimGen:

    def __init__(self, posegen: PositionGen, airsim: AirSimClient):
        self.airsim = airsim
        self.pose_gen = posegen

    def generate(self, n_samples) -> [(Image, ImgLabel)]:
        samples = []
        labels = []
        self.airsim.reset()
        for i in range(n_samples):
            pose = self.pose_gen.gen_pos()
            self.airsim.setPose(pose)
            img, label = self.airsim.retrieve_samples()
            samples.append(img)
            labels.append(label)

        return samples,labels
